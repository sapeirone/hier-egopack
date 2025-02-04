#
# Single task training.
# 

import os
import logging
import warnings

import hydra
import omegaconf

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_  # type: ignore

from data.base_dataset import BaseDataset

from models.tasks.task import Task

import wandb

from train_egopack import enable_gradients
from utils.dataloading import InfiniteLoader, build_dataloader
from torch_geometric.loader import DataLoader
from utils.hash import compute_hash
from utils.wandb import format_run_name
from utils.random import seed_everything
from utils.optimizers import build_optimizer

from typing import Optional

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
    task: Task, 
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
        temporal model (e.g. TRN or GraphUNet)
    optimizer : torch.optim.Optimizer
        optimizer for the parameters
    warmup_scheduler : Optional[torch.optim.lr_scheduler.LRScheduler]
        warmup scheduler
    lr_warmup_steps : int
        number of warmup steps
    scheduler : Optional[torch.optim.lr_scheduler.LRScheduler]
        scheduler (post-warmup)
    dataloader : Iterator[Data]
        infinite dataloader over the input data
    task : Task
        main task
    meter: TrainMeter
        meter for training metrics
    gradient_clip : float, optional
        gradient clip value, by default -1.0
    num_steps : int, optional
        number of training steps to perform, by default 1000
    device : str, optional
        device, by default "cuda"
    """
    meter.reset()
    mean_loss_meter = MeanMetric().to(device)

    # Put models in train mode
    model.train()
    task.train()
    
    pbar = tqdm(range(start_step, start_step + num_steps), desc="Training...", leave=False)
    for step in pbar:
        (data, ) = next(dataloader)  # type: ignore
        
        # Processing using the temporal graph
        _, graphs = model(data.to(device=device))
        outputs = task(graphs, data)
        
        # Compute the intermediate losses for each layer in the UNet
        loss, losses = task.compute_loss(outputs, graphs, data)

        meter.update(losses)
        mean_loss_meter.update(loss)

        # Backpropagate the sum of all the tasks' losses
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
        
    logger.info(f"Training loss = {mean_loss_meter.compute():.4f}.")
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

        # Processing using the temporal graph
        _, graphs = model(data.to(device=device))
        # Task input: (features, pos, batch, segments, segments_batch)
        outputs = task(graphs, data)
        
        # Compute the intermediate losses for each layer in the UNet
        loss, losses = task.compute_loss(outputs, graphs, data)
        meter.update(outputs, losses, graphs, data)
        
        pbar.set_description(f"Validation loss = {loss.mean().item():.4f}.")
        
    meter.logs(artifacts_path)
    meter.wandb_logs(val_step)


@hydra.main(config_path="configs/experiments/single_tasks/", version_base="1.3")
def main(cfg):
    
    config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run_name, run_name_without_date = format_run_name(cfg.name_pattern, cfg)
    
    logger.info("Run name: %s", run_name)
    
    wandb.init(name=run_name, config=config, anonymous='must')  # type: ignore

    rng_generator = seed_everything(cfg.seed)

    ########################################
    # 1. Initialize moment queries dataset #
    ########################################
    logger.info("")
    logger.info("Setting up Ego4d Action Recognition datasets...")
    dset_train: BaseDataset = hydra.utils.instantiate(cfg.dataset, split="train", features=cfg.features)
    dset_val: BaseDataset = hydra.utils.instantiate(cfg.dataset, split="val", features=cfg.features)

    dl_train: DataLoader = build_dataloader(dset_train, cfg.batch_size, True, cfg.num_workers, drop_last=True, rng_generator=rng_generator)
    dl_val: DataLoader = build_dataloader(dset_val, 1 if 'mq' in dset_val.name else cfg.batch_size, False, cfg.num_workers, drop_last=False, rng_generator=rng_generator)

    inf_dl_train: InfiniteLoader = iter(InfiniteLoader([dl_train]))

    ################################
    # 2. Initialize model and task #
    ################################
    model = hydra.utils.instantiate(cfg.model, input_size=dset_train.features.size, _recursive_=False).to(cfg.device)
    task = hydra.utils.instantiate(cfg.task, _recursive_=False).to(cfg.device)

    # Log gradients and parameters for both the temporal graph model and the available tasks
    wandb.run.watch(model, log="all", log_freq=100, idx=0, log_graph=False)  # type: ignore
    wandb.run.watch(task, log="all", log_freq=100, idx=1, log_graph=False)  # type: ignore
        
    ###########################################
    # 3. Initialize optimizers and schedulers #
    ###########################################

    # Build the optimizer
    enable_gradients([model], cfg.train_temporal_model)
    optimizer, n_trainable_params = build_optimizer([m for m in [model, task] if m is not None], cfg.optimizer)
    wandb.run.summary['trainable_params'] = n_trainable_params  # type: ignore

    logger.info(f"Setting up the learning rate scheduler...")

    steps_per_round = cfg.steps_per_round
    warmup_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    scheduler = None
    if cfg.lr_warmup:
        assert cfg.lr_warmup_epochs < cfg.num_epochs, "Warmup steps should be less than the total number of epochs."
        start_alpha, end_alpha = 0.001, 1
        logger.info(f"Using lr warmup for {cfg.lr_warmup_epochs} epochs from {start_alpha} to {end_alpha}.")
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_alpha, end_alpha, cfg.lr_warmup_epochs * len(dl_train))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (cfg.num_epochs - cfg.lr_warmup_epochs) * len(dl_train), cfg.lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.num_epochs * len(dl_train), cfg.lr_min)
        
    ####################################
    # 3. Starting the training process #
    ####################################
    
    # Setup meters and wandb
    wandb.define_metric("train/step", summary='none')
    wandb.define_metric("val/step", summary='none')
    train_meter = hydra.utils.instantiate(cfg.train_meter, prefix=task.name, step_metric="train/step", device=cfg.device)
    eval_meter = hydra.utils.instantiate(cfg.eval_meter, prefix=task.name, step_metric="val/step", device=cfg.device)
    
    train_meter.define_metrics()
    eval_meter.define_metrics()

    train_step = 0
    
    if cfg.from_pretrain is not None and cfg.resume_from is not None:
        raise ValueError("Cannot resume from a checkpoint and load a pretrained model at the same time.")
    
    if cfg.from_pretrain:
        logger.info(f"Loading pretrained temporal backbone from {cfg.from_pretrain}...")
        logger.info(f"Hash: {compute_hash(cfg.from_pretrain, 'sha256')}")

        state = torch.load(cfg.from_pretrain, map_location=cfg.device)
        model.load_state_dict(state['model'], strict=False)
    
    if cfg.resume_from:
        resume_from = cfg.resume_from
        if cfg.resume_from.endswith(".latest"):
            basename = os.path.basename(resume_from)
            resume_from = os.path.join(os.path.dirname(resume_from), sorted([f for f in os.listdir(os.path.dirname(cfg.resume_from)) if f.startswith(basename) and f.endswith('.pth')])[-1])
        logger.info(f"Resuming from {resume_from}...")
        logger.info(f"Hash: {compute_hash(resume_from, 'sha256')}")

        state = torch.load(resume_from, map_location=cfg.device)
        model.load_state_dict(state['model'])
        task.load_state_dict(state['task'])
        optimizer.load_state_dict(state['optimizer'])
        if warmup_scheduler is not None:
            warmup_scheduler.load_state_dict(state['warmup_scheduler'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        train_step = state['step']

    if cfg.train:
        max_steps = cfg.num_epochs * len(dl_train)
        
        for train_step in range(train_step, max_steps, steps_per_round):
            val_step = train_step // steps_per_round
            
            logger.info("")
            logger.info(f"Starting training step n. {train_step:3d}/{cfg.num_epochs * len(dl_train)}...")

            train(
                train_step,
                model,
                optimizer,
                warmup_scheduler,
                cfg.lr_warmup_epochs * len(dl_train) if cfg.lr_warmup else 0,
                scheduler,
                inf_dl_train,
                task,
                meter=train_meter,
                gradient_clip=cfg.gradient_clip,
                num_steps=min(steps_per_round, max_steps - train_step),
                device=cfg.device,
            )

            if train_step and (cfg.eval_interval > 0) and (train_step % cfg.eval_interval == 0):
                logger.info(f"Starting validation process for step {train_step:3d}/{cfg.num_epochs * len(dl_train)}...")
                validate(val_step, model, dl_val, task, meter=eval_meter, device=cfg.device)  # type: ignore
                logger.info("")
            
            if cfg.save_to is not None:
                path = f"{cfg.save_to}/{run_name}.pth"
                logger.info(f"Saving model to {path}...")
                os.makedirs(cfg.save_to, exist_ok=True)
                torch.save({
                    "model": model.state_dict(),
                    "task": task.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "warmup_scheduler": warmup_scheduler.state_dict() if warmup_scheduler is not None else None,
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "step": train_step + steps_per_round,
                    "config": cfg
                }, path)
                logger.info(f"Hash: {compute_hash(path, 'sha256')}")

    logger.info("")
    logger.info("Starting final validation...")
    val_step = train_step // steps_per_round
    validate(val_step, model, dl_val, task, meter=eval_meter, device=cfg.device,
             artifacts_path=f"{cfg.save_to}/{run_name}" if cfg.save_to else None)  # type: ignore


if __name__ == "__main__":
    main()
    wandb.finish()
