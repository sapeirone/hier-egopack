import hydra
import torch

from typing import List, Any, Tuple

import logging
logger = logging.getLogger(__name__)


class OptimizerCfg:
    _target_: Any  # hydra instantiation target
    lr: float
    weight_decay: float


def build_optimizer(models: List[torch.nn.Module], cfg: OptimizerCfg, verbose: bool = True) -> Tuple[torch.optim.Optimizer, float]:
    """Build optimizer for a list of nn.Module modules.

    Parameters
    ----------
    models : List[torch.nn.Module]
        the list of models whose parameters should be part of the optimizer
    cfg : OptimizerCfg
        the configuration of the optimizer

    Returns
    -------
    Tuple[torch.optim.Optimizer, float]
        the instantiated optimizer and the number of trainable parameters (in millions)
    """
    # Collect all parameters and set up the optimizer
    parameters = [param for m in models for param in m.parameters() if param.requires_grad]
    optimizer = hydra.utils.instantiate(cfg, [
        {'params': [p for p in parameters if len(p.shape) == 1], 'lr': cfg.lr, 'weight_decay': 0},
        {'params': [p for p in parameters if len(p.shape) > 1], 'lr': cfg.lr, 'weight_decay': cfg.weight_decay}
    ])
    
    nparams = sum(p.numel() for pg in optimizer.param_groups for p in pg['params']) / 1e6
    if verbose:
        logger.info(f"Model has {nparams:.4f}M trainable parameters.")
        
    return optimizer, nparams