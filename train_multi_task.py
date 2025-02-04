#
# Multi task training.
# 

import logging
import warnings
import os

import hydra
import omegaconf

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_  # type: ignore

from data.base_dataset import BaseDataset

from models.tasks.task import Task

import wandb

from utils.dataloading import InfiniteLoader, build_dataloader
from torch_geometric.loader import DataLoader
from utils.hash import compute_hash
from utils.wandb import format_run_name
from utils.random import seed_everything
from utils.optimizers import build_optimizer

from typing import Optional, Dict

from tqdm.auto import tqdm

from torchmetrics.aggregation import MeanMetric
from utils.meters.ego4d import BaseMeter as Meter


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# TODO: temporary fix for torch geometric warning
warnings.filterwarnings("ignore")


def train(
    start_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    warmup_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    lr_warmup_steps: int,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    dataloader: InfiniteLoader,
    tasks: Dict[str, Task],
    meters: Dict[str, Meter],
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
        temporal model (e.g. TRN or GraphUNet)
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
    tasks : List[Task]
        main task
    meters : List[TrainMeter]
        meter for training metrics
    gradient_clip : float, optional
        gradient clip value, by default -1.0
    num_steps : int, optional
        number of training steps to perform, by default 1000
    device : str, optional
        device, by default "cuda"
    """
    mean_loss_meters = {task: MeanMetric().to(device) for task in tasks.keys()}

    # Put models in train mode
    model.train()
    
    for name in tasks.keys():
        tasks[name].train()
        meters[name].reset()
    
    pbar = tqdm(range(start_step, start_step + num_steps), desc="Training...", leave=False)
    for step in pbar:
        
        # Backpropagate the sum of all the tasks' losses
        optimizer.zero_grad()
        data = next(dataloader)  # type: ignore
        
        total_loss = 0
        
        for name in tasks.keys():
            task = tasks[name]
            task_meter = meters[name]
            task_data = data[name]  # type: ignore
            
            # Processing using the temporal graph
            _, graphs = model(task_data.to(device=device))
            outputs = task(graphs, task_data)
            
            # Compute the intermediate losses for each layer in the UNet
            loss, losses = task.compute_loss(outputs, graphs, task_data)

            task_meter.update(losses)
            task_meter.wandb_logs(step, additional_logs={"lr": optimizer.param_groups[0]["lr"]})
            task_meter.reset()
            
            mean_loss_meters[name].update(loss)
            
            total_loss += loss.mean().detach().item()

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
        
        
    logger.info(f"Training losses: " + ", ".join(f"{name}: {m.compute():.4f}" for name, m in mean_loss_meters.items()) + ".")
    logger.info("")


@torch.no_grad()
def validate(
    val_step: int,
    model: nn.Module,
    dataloader: DataLoader,
    task: Task,
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

    # Put models in eval mode
    model.eval()
    task.eval()
    
    meter.reset()

    pbar = tqdm(dataloader, desc="Validating...", leave=False)
    for data in pbar:
        data = data.to(device=device)

        # Processing using the temporal graph
        _, graphs = model(data)
        # Task input: (features, pos, batch, segments, segments_batch)
        outputs = task(graphs, data)
        
        # Compute the intermediate losses for each layer in the UNet
        loss, losses = task.compute_loss(outputs, graphs, data)
        meter.update(outputs, losses, graphs, data)
        
        pbar.set_description(f"Validation loss = {loss.mean().item():.4f}.")
        
    meter.logs(artifacts_path)
    meter.wandb_logs(val_step)


@hydra.main(config_path="configs/experiments/multi_tasks", version_base="1.3")
def main(cfg):
    
    config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run_name, run_name_without_date = format_run_name(cfg.name_pattern, cfg)
    
    logger.info("Run name: %s", run_name)
    
    wandb.init(name=run_name, config=config, anonymous='must')  # type: ignore

    rng_generator = seed_everything(cfg.seed)

    ##########################
    # 1. Initialize datasets #
    ##########################
    logger.info("")
    logger.info("Setting up training datasets...")
    dsets_train: Dict[str, BaseDataset] = {task: hydra.utils.instantiate(cfg.datasets[task], split="train", features=cfg.features) for task in cfg.mtl_tasks}
    
    logger.info("Dataset sizes: " + ", ".join(f"{name}: {len(dset)}" for name, dset in dsets_train.items()) + ".")
    
    logger.info("")
    logger.info("Setting up validation datasets...")
    dsets_val: Dict[str, BaseDataset] = {task: hydra.utils.instantiate(cfg.datasets[task], split="val", features=cfg.features) for task in cfg.mtl_tasks}

    logger.info("Dataset sizes: " + ", ".join(f"{name}: {len(dset)}" for name, dset in dsets_val.items()) + ".")

    # Find the smallest dataset to scale the batch size accordingly
    min_dset_length = min(len(dset) for dset in dsets_train.values())
    dataloaders_train: Dict[str, DataLoader] = {
        task: build_dataloader(dset, cfg.batch_size * (len(dset) // min_dset_length), True, cfg.num_workers, drop_last=True, rng_generator=rng_generator) 
        for task, dset in dsets_train.items()
    }
    
    # Find the smallest dataset to scale the batch size accordingly
    min_dset_length = min(len(dset) for dset in dsets_val.values())
    dataloaders_val: Dict[str, DataLoader] = {
        task: build_dataloader(dset, 1 if 'mq' in task else cfg.batch_size * (len(dset) // min_dset_length), False, cfg.num_workers, drop_last=False, rng_generator=rng_generator) 
        for task, dset in dsets_val.items()
    }

    infinite_dl_train: InfiniteLoader = iter(InfiniteLoader(dataloaders_train))

    ################################
    # 2. Initialize model and task #
    ################################
    features_size = dsets_train[cfg.mtl_tasks[0]].features.size
    assert all(dset.features.size == features_size for dset in dsets_train.values())
    model = hydra.utils.instantiate(cfg.model, input_size=features_size, _recursive_=False).to(cfg.device)
    tasks: Dict[str, Task] = {task: hydra.utils.instantiate(cfg.tasks[task], _recursive_=False).to(cfg.device) for task in cfg.mtl_tasks}

    # Log gradients and parameters for both the temporal graph model and the available tasks
    for idx, m in enumerate([model, *tasks]):
        wandb.run.watch(model, log="all", log_freq=100, idx=idx, log_graph=False)  # type: ignore
        
    ###########################################
    # 3. Initialize optimizers and schedulers #
    ###########################################

    # Build the optimizer
    optimizer, n_trainable_params = build_optimizer([m for m in [model, *tasks.values()] if m is not None], cfg.optimizer)
    wandb.run.summary['trainable_params'] = n_trainable_params  # type: ignore

    logger.info(f"Setting up the learning rate scheduler...")

    steps_per_round = cfg.steps_per_round
    warmup_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    scheduler = None
    if cfg.lr_warmup:
        assert cfg.lr_warmup_epochs < cfg.num_epochs, "Warmup steps should be less than the total number of epochs."
        start_alpha, end_alpha = 0.001, 1
        logger.info(f"Using lr warmup for {cfg.lr_warmup_epochs} epochs from {start_alpha} to {end_alpha}.")
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_alpha, end_alpha, cfg.lr_warmup_epochs * len(infinite_dl_train))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (cfg.num_epochs - cfg.lr_warmup_epochs) * len(infinite_dl_train), 0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.num_epochs * len(infinite_dl_train), 0)
        
    ####################################
    # 3. Starting the training process #
    ####################################
    
    # Setup meters and wandb
    wandb.run.define_metric("train/step", summary='none')  # type: ignore
    wandb.run.define_metric("val/step", summary='none')  # type: ignore
    train_meters = {
        task: hydra.utils.instantiate(cfg.train_meters[task], prefix=task, step_metric="train/step", device=cfg.device) 
        for task in cfg.mtl_tasks
    }
    eval_meters = {
        task: hydra.utils.instantiate(cfg.eval_meters[task], prefix=task, step_metric="val/step", device=cfg.device) 
        for task in cfg.mtl_tasks
    }
    
    for meter in [*train_meters.values(), *eval_meters.values()]:
        meter.define_metrics()
    
    train_step = 0
    
    if cfg.resume_from:
        resume_from = cfg.resume_from
        if resume_from.endswith(".latest"):
            basename = os.path.basename(resume_from).replace('.latest', '')
            resume_from = os.path.join(os.path.dirname(resume_from), sorted([f for f in os.listdir(os.path.dirname(cfg.resume_from)) if f.startswith(basename) and f.endswith('.pth')])[-1])

        logger.info(f"Resuming from {resume_from}...")
        logger.info(f"Hash: {compute_hash(resume_from, 'sha256')}")

        state = torch.load(resume_from, map_location=cfg.device)
        model.load_state_dict(state['model'])
        for name, task in tasks.items():
            task.load_state_dict(state['task'][name])
        optimizer.load_state_dict(state['optimizer'])
        if warmup_scheduler is not None:
            warmup_scheduler.load_state_dict(state['warmup_scheduler'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        train_step = state['step']

    if cfg.train:
        max_steps = cfg.num_epochs * len(infinite_dl_train)
        for val_step, train_step in enumerate(range(0, max_steps, steps_per_round)):
            logger.info("")
            logger.info(f"Starting training step n. {train_step:3d}/{cfg.num_epochs * len(infinite_dl_train)}...")

            train(
                train_step,
                model,
                optimizer,
                warmup_scheduler,
                cfg.lr_warmup_epochs * len(infinite_dl_train) if cfg.lr_warmup else 0,
                scheduler,
                infinite_dl_train,
                tasks,
                meters=train_meters,
                gradient_clip=cfg.gradient_clip,
                num_steps=min(steps_per_round, max_steps - train_step),
                device=cfg.device,
            )

            if train_step and (cfg.eval_interval > 0) and (train_step % cfg.eval_interval == 0):
                for name in tasks.keys():
                    logger.info(f"Starting validation process for task {name} for step {train_step:3d}/{cfg.num_epochs * len(infinite_dl_train)}...")
                    validate(val_step, model, dataloaders_val[name], tasks[name], meter=eval_meters[name], device=cfg.device)
                    logger.info("")
                    
            if cfg.save_to is not None:
                path = f"{cfg.save_to}/{run_name}.pth"
                logger.info(f"Saving model to {path}...")
                os.makedirs(cfg.save_to, exist_ok=True)
                torch.save({
                    "model": model.state_dict(),
                    "task": {name: task.state_dict() for name, task in tasks.items()},
                    "optimizer": optimizer.state_dict(),
                    "warmup_scheduler": warmup_scheduler.state_dict() if warmup_scheduler is not None else None,
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "step": train_step + steps_per_round,
                    "config": cfg
                }, path)
                logger.info(f"Hash: {compute_hash(path, 'sha256')}")

    logger.info("Training completed! Starting final validation...")
    for name in tasks.keys():
        logger.info(f"Starting validation process for task {name} for step {train_step:3d}/{cfg.num_epochs * len(infinite_dl_train)}...")
        validate(val_step, model, dataloaders_val[name], tasks[name], meter=eval_meters[name], 
                 device=cfg.device, artifacts_path=f"{cfg.save_to}/{run_name}.pth" if cfg.save_to else None)
        logger.info("")


if __name__ == "__main__":
    main()
    wandb.run.finish()  # type: ignore
