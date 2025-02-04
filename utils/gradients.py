"""Utils for gradients manipulation"""

from typing import List

from torch import nn


def enable_gradients(models: List[nn.Module], enabled: bool = True):
    """Recursively enable (or disable) gradients for a list of modules.

    Parameters
    ----------
    models : List[nn.Module]
        list of models
    enabled : bool, optional
        whether to enable or disable gradients computation, by default True
    """
    for model in models:
        for p in model.parameters():
            p.requires_grad_(enabled)
