"""Main file for EgoPack training"""

import logging
import warnings
from typing import Optional, Dict, Any

import os
from os import path as osp
import hydra
import omegaconf

from tqdm.auto import tqdm

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_  # type: ignore

from torchmetrics.aggregation import MeanMetric

from torch_geometric.utils import scatter
from torch_geometric.loader import DataLoader

import wandb

from data.base_dataset import BaseDataset

from utils.dataloading import InfiniteLoader, build_dataloader
from utils.gradients import enable_gradients
from utils.hash import compute_hash
from utils.wandb import format_run_name
from utils.random import seed_everything
from utils.optimizers import build_optimizer
from utils.meters.ego4d import BaseMeter as Meter

from models.egopack import EgoPack
from models.tasks.task import Task


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.filterwarnings("ignore")


@torch.inference_mode()
def build_prototypes(
    model: nn.Module,
    dataloader: DataLoader,
    alignment_task: Task,
    tasks: Dict[str, Task],
    device: torch.device = torch.device("cpu"),
):
    """Build prototypes for EgoPack.

    Parameters
    ----------
    model : nn.Module
        temporal model (e.g. TRN or GraphUNet)
    dataloader : DataLoader
        dataloader over the input data
    tasks : Dict[str, Task]
        dictionary of tasks
    device : str, optional
        device, by default "cuda"
    """
    # Put all the models and tasks in eval mode
    for m in [model, alignment_task, *tasks.values()]:
        m.eval()

    # Since we are using action labels to extrapolate the prototypes, we initialize as many prototypes as there are action classes
    num_verbs, num_nouns = (alignment_task.verb_classifier.cls.out_features), (alignment_task.noun_classifier.cls.out_features)  # type: ignore
    size: int = num_verbs * num_nouns  # type: ignore
    feat_size = model.hidden_size

    prototypes = {task: torch.zeros((size, feat_size), device=device) for task in tasks.keys()}  # type: ignore
    all_labels = []

    for data in dataloader:

        # Processing using the temporal graph up to the first layer
        _, graphs = model(data.to(device=device), max_depth=0)
        
        # hard assignment using labels
        all_labels.append(labels := data.verb_labels * num_nouns + data.noun_labels)
        
        temp_feat = graphs.x.clone()

        # collect the different point of views of the different tasks
        for task in tasks.values():
            # Align features according to the boundaries of the action recognition task
            graphs.x = task.project(temp_feat)
            
            task_feat = alignment_task.align(graphs, data)  # type: ignore
            
            prototypes[task.name] = prototypes[task.name] + scatter(task_feat, labels, dim_size=size, reduce="sum")

        del graphs, temp_feat, task_feat

    # Keep only the prototypes that have a non-empty support
    support = torch.cat(all_labels).bincount(minlength=size).float()
    prototypes = {name: (task_prototypes[support > 0] / support[support > 0, None]).float() for name, task_prototypes in prototypes.items()}
    
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    
    # Compute the verb and noun labels for the prototypes
    verb_labels = torch.arange(0, int(size), dtype=int, device=support.device)[support > 0] // num_nouns  # type: ignore
    noun_labels = torch.arange(0, int(size), dtype=int, device=support.device)[support > 0] % num_nouns  # type: ignore
    
    return prototypes, verb_labels, noun_labels


def train(
    start_step: int,
    model: nn.Module,
    egopack: EgoPack,
    optimizer: Optimizer,
    warmup_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    lr_warmup_steps: int,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    dataloader: InfiniteLoader,
    main_task: Task,
    aux_tasks: Dict[str, Task],
    meter: Meter,
    gradient_clip: float = -1.0,
    num_steps: int = 1000,
    device: torch.device = torch.device("cpu"),
):
    """Run a number of training steps.

    Parameters
    ----------
    start_step : int
        Initial training step
    model : nn.Module
        temporal model (e.g. TRN or GraphHier)
    egopack: EgoPack
        EgoPack instance
    optimizer : torch.optim.Optimizer
        optimizer for the parameters
    warmup_scheduler : Optional[torch.optim.lr_scheduler.LRScheduler]
        warmup scheduler
    lr_warmup_steps : int
        number of warmup steps
    scheduler : Optional[torch.optim.lr_scheduler.LRScheduler]
        scheduler (post-warmup)
    dataloaders : InfiniteLoader
        infinite dataloader over the input data
    main_task: Task
        primary task
    tasks : List[Task]
        main task
    meter : TrainMeter
        meter for training metrics
    gradient_clip : float, optional
        gradient clip value, by default -1.0
    num_steps : int, optional
        number of training steps to perform, by default 1000
    device : str, optional
        device, by default "cuda"
    """
    loss_meter = MeanMetric().to(device)
    
    # Put all the models, tasks and EgoPack in train mode
    for m in [model, egopack, main_task, *aux_tasks.values()]:
        m.train()
    
    pbar = tqdm(range(start_step, start_step + num_steps), desc="Training...", leave=False)
    for step in pbar:
        (data, ) = next(dataloader)  # type: ignore
        
        # Step 1: Process data through the temporal backbone
        _, graphs = model(data.to(device=device))
        
        # Step 2: Project the output of the temporal backbone into the features space of each auxiliary task
        temporal_features = graphs.x.detach()  # No gradient to the temporal backbone from the auxiliary tasks
        pos, video, depth = graphs.pos, graphs.video, graphs.depth
        aux_features = {name: task.project(temporal_features) for name, task in aux_tasks.items()}
        # Align the auxiliary features to the main task
        aux_features = {name: main_task.align_for_egopack(task_feat, pos, video, depth, data) for name, task_feat in aux_features.items()}
        # Task interaction with EgoPack
        aux_features, _ = egopack.interact(aux_features)

        # Step 3: Compute task output by combining the main task features with auxiliary tasks
        outputs = main_task(graphs, data, aux_features)
        
        # Step 4: loss computation
        loss, losses = main_task.compute_loss(outputs, graphs, data)
        meter.update(losses)
        loss_meter.update(loss)

        optimizer.zero_grad()
        loss.mean().backward()

        # Clip gradients (eventually)
        if gradient_clip > 0:
            clip_grad_norm_([p for pg in optimizer.param_groups for p in pg["params"]], gradient_clip)
        optimizer.step()

        # Update the learning rate according to the warmup scheduler
        if warmup_scheduler is not None and step <= lr_warmup_steps:
            warmup_scheduler.step(step - 1)

        # Update the learning rate according to the scheduler
        if scheduler is not None and step > lr_warmup_steps:
            scheduler.step(step - 1 - lr_warmup_steps)

        pbar.set_description(f"Training step {step} (loss = {loss.mean().item():.4f}).")
        
        meter.wandb_logs(step, additional_logs={"lr": optimizer.param_groups[0]["lr"]})
        meter.reset()
        
    logger.info("Average training loss = %.4f.", loss_meter.compute())
    logger.info("")


@torch.no_grad()
def validate(
    val_step: int,
    model: nn.Module,
    egopack: EgoPack,
    main_task: Task,
    aux_tasks: Dict[str, Task],
    dataloader: DataLoader,
    meter: Meter,
    device: torch.device = torch.device("cpu"),
    artifacts_path: Optional[str] = None,
):
    """Run a validation step.

    Parameters
    ----------
    val_step : int
        current validation step
    model : nn.Module
        temporal model (e.g. TRN or GraphUNet)
    dataloader : DataLoader
        dataloader over validation data
    task : Task
        main task
    meter: Meter
        meter for evaluation metrics
    device : str, optional
        device, by default "cuda"
    """

    # Put all the models, tasks and EgoPack in train mode
    for m in [model, main_task, egopack, *aux_tasks.values()]:
        m.eval()
    
    meter.reset()

    pbar = tqdm(dataloader, desc="Validating...", leave=False)
    for data in pbar:

        # Processing using the temporal graph
        _, graphs = model(data.to(device=device))
        
        # Collect outputs of the secondary tasks (to be used for prototypes matching in EgoPack)
        temporal_features = graphs.x.detach()  # No gradient to the temporal backbone from the auxiliary tasks
        pos, video, depth = graphs.pos, graphs.video, graphs.depth
        aux_features = {name: task.project(temporal_features) for name, task in aux_tasks.items()}
        # Align the auxiliary features to the main task
        aux_features = {name: main_task.align_for_egopack(task_feat, pos, video, depth, data) for name, task_feat in aux_features.items()}
        # Task interaction with EgoPack
        aux_features, _ = egopack.interact(aux_features)

        outputs = main_task(graphs, data, aux_features)
        
        # Compute the intermediate losses for each layer in the UNet
        loss, losses = main_task.compute_loss(outputs, graphs, data)
        meter.update(outputs, losses, graphs, data)
        
        pbar.set_description(f"Validation loss = {loss.mean().item():.4f}.")
        
    meter.logs(artifacts_path)
    meter.wandb_logs(val_step)


@hydra.main(config_path="configs/experiments/egopack/", version_base="1.3")
def main(cfg: Any):
    """EgoPack training entrypoint.

    Parameters
    ----------
    cfg : Dict[Any, Any]
        configuration from hydra
    """
    
    # Hydra configuration setup
    config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run_name, run_name_without_date = format_run_name(cfg.name_pattern, cfg)
    
    if run_name is None:
        logger.warning("No name provided for this run.")
    else:
        logger.info("Run name: %s", run_name)
    
    wandb.init(name=run_name, config=config, anonymous='must')  # type: ignore

    rng_generator = seed_everything(cfg.seed)

    #####################################################
    # 1. Initialize the dataset for the downstream task #
    #####################################################
    logger.info("")
    logger.info("Setting up datasets...")
    dset_train: BaseDataset = hydra.utils.instantiate(cfg.datasets[cfg.egopack_task.name], split="train", features=cfg.features)
    dset_val: BaseDataset = hydra.utils.instantiate(cfg.datasets[cfg.egopack_task.name], split="val", features=cfg.features)
    
    if cfg.train_on_val:
        logger.warning("!!!!!! TRAINING ALSO ON THE VALIDATION SET !!!!!!")
        dl_train = torch.utils.data.ConcatDataset([dset_train, dset_val])  # type: ignore

    dl_train: DataLoader = build_dataloader(dset_train, cfg.batch_size, True, cfg.num_workers, drop_last=True, rng_generator=rng_generator)
    dl_val: DataLoader = build_dataloader(dset_val, 1 if 'mq' in dset_val.name else cfg.batch_size, False, cfg.num_workers, 
                                          drop_last=False, rng_generator=rng_generator)

    inf_dl_train: InfiniteLoader = iter(InfiniteLoader([dl_train]))

    ################################
    # 2. Initialize model and task #
    ################################
    
    resume_from = cfg.resume_from
    if resume_from.endswith(".latest"):
        basename = osp.basename(resume_from).replace('.latest', '')
        file = sorted([f for f in os.listdir(osp.dirname(cfg.resume_from)) if f.startswith(basename) and f.endswith('.pth')])[-1]
        resume_from = osp.join(osp.dirname(resume_from), file)

    logger.info("Resuming from %s...", resume_from)
    logger.info("Hash: %s", compute_hash(resume_from, 'sha256'))
    
    state = torch.load(resume_from, map_location=cfg.device)
    mtl_tasks = list(state['task'].keys())
    model = hydra.utils.instantiate(cfg.model, input_size=dset_train.features.size, _recursive_=False).to(cfg.device)
    tasks: Dict[str, Task] = {task: hydra.utils.instantiate(cfg.tasks[task], _recursive_=False).to(cfg.device) for task in mtl_tasks}
    
    alignment_task = hydra.utils.instantiate(cfg.tasks["ego4d/ar"], _recursive_=False).to(cfg.device)
    egopack_task = hydra.utils.instantiate(cfg.egopack_task, aux_tasks=mtl_tasks, _recursive_=False).to(cfg.device)
        
    model.load_state_dict(state['model'], strict=False)
    for name, task in tasks.items():
        task.load_state_dict(state['task'][name])
        
    ###################################
    # 3. Build the EgoPack prototypes #
    ###################################
    logger.info("Building prototypes from tasks [%s]...", ', '.join(tasks.keys()))
    dset_prototypes: Dict[str, BaseDataset] = hydra.utils.instantiate(cfg.datasets["ego4d/ar"], split="train", features=cfg.features)
    dl = build_dataloader(dset_prototypes, 8, True, cfg.num_workers, drop_last=False)
    prototypes, *_ = build_prototypes(model, dl, alignment_task, tasks, device=cfg.device)
    egopack = EgoPack(prototypes, model.hidden_size, **cfg.egopack).to(cfg.device)
    
    ###########################################
    # 4. Initialize optimizers and schedulers #
    ###########################################
    
    modules = [model, *tasks.values(), egopack_task, egopack]
    
    # Log gradients and parameters for both the temporal graph model and the available tasks
    for idx, m in enumerate(modules):
        wandb.run.watch(m, log="all", log_freq=100, idx=idx, log_graph=False)  # type: ignore

    # Train (or not) the temporal model and the projections
    enable_gradients([model], cfg.train_temporal_model)
    enable_gradients(list(tasks.values()), cfg.train_projections)
    optimizer, n_trainable_params = build_optimizer(modules, cfg.optimizer)
    wandb.run.summary['trainable_params'] = n_trainable_params  # type: ignore

    logger.info("Setting up the learning rate scheduler...")

    steps_per_round = cfg.steps_per_round
    warmup_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    scheduler = None
    if cfg.lr_warmup:
        assert cfg.lr_warmup_epochs < cfg.num_epochs, "Warmup steps should be less than the total number of epochs."
        start_alpha, end_alpha = 0.001, 1
        logger.info("Using lr warmup for %d epochs from %f to %f.", cfg.lr_warmup_epochs, start_alpha, end_alpha)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_alpha, end_alpha, cfg.lr_warmup_epochs * len(inf_dl_train))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (cfg.num_epochs - cfg.lr_warmup_epochs) * len(inf_dl_train), cfg.lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.num_epochs * len(inf_dl_train), cfg.lr_min)
        
    ####################################
    # 3. Starting the training process #
    ####################################
    
    # Setup meters and wandb
    wandb.define_metric("train/step", summary='none')
    wandb.define_metric("val/step", summary='none')
    train_meter = hydra.utils.instantiate(cfg.train_meters[cfg.egopack_task.name], prefix=cfg.egopack_task.name, step_metric="train/step", device=cfg.device)
    eval_meter = hydra.utils.instantiate(cfg.eval_meters[cfg.egopack_task.name], prefix=cfg.egopack_task.name, step_metric="val/step", device=cfg.device)
    
    train_meter.define_metrics()
    eval_meter.define_metrics()
    
    logger.info("")

    max_steps = cfg.num_epochs * len(inf_dl_train)
    for val_step, train_step in enumerate(range(0, max_steps, steps_per_round)):
        logger.info("Starting training step n. %d/%d...", train_step, cfg.num_epochs * len(inf_dl_train))

        train(
            train_step,
            model,
            egopack,
            optimizer,
            warmup_scheduler,
            cfg.lr_warmup_epochs * len(inf_dl_train) if cfg.lr_warmup else 0,
            scheduler,
            inf_dl_train,
            egopack_task,
            tasks,
            meter=train_meter,
            gradient_clip=cfg.gradient_clip,
            num_steps=min(steps_per_round, max_steps - train_step),
            device=cfg.device,
        )

        if train_step and (cfg.eval_interval > 0) and (train_step % cfg.eval_interval == 0):
            logger.info("Starting validation process for step %d/%d...", train_step, cfg.num_epochs * len(dl_train))
            validate(val_step, model, egopack, egopack_task, tasks, dl_val, eval_meter, device=cfg.device)
            logger.info("")
            
    if cfg.save_to is not None:
        path = f"{cfg.save_to}/{run_name}.pth"
        logger.info("Saving model to %s...", path)
        os.makedirs(cfg.save_to, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "egopack_task": egopack_task.state_dict(),
            "tasks": {name: task.state_dict() for name, task in tasks.items()},
            "egopack": egopack.state_dict(),
            "optimizer": optimizer.state_dict(),
            "warmup_scheduler": warmup_scheduler.state_dict() if warmup_scheduler is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": train_step + steps_per_round,
            "config": cfg
        }, path)
        logger.info("Hash: %s", compute_hash(path, 'sha256'))

    logger.info("")
    logger.info("Starting final validation on EMA model...")
    val_step = train_step // steps_per_round
    validate(val_step, model, egopack, egopack_task, tasks, dl_val, eval_meter, device=cfg.device,
             artifacts_path=f"{cfg.save_to}/{run_name}.pth" if cfg.save_to else None)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
    wandb.finish()
