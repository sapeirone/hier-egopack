import random
import numpy as np

import os
import logging

import torch
import torch.backends.cudnn as cudnn

logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    """
    Seeds everything with the given seed.
    """
    logger.info(f"Seeding everything with seed {seed}.")

    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # training: disable cudnn benchmark to ensure the reproducibility
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # this is needed for CUDA >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    return rng_generator
