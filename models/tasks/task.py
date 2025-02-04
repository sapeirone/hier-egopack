"""Base classes for Single Task and EgoPack models"""

import logging
from typing import Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from models.tasks.utils import Projection


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Task(torch.nn.Module):

    # pylint: disable=unused-argument
    def __init__(self, name: str, input_size: int, features_size: int, dropout: float = 0, **kwargs):
        super().__init__()
        self.name = name
        
        self.projector = self.build_projection(input_size, features_size, dropout)
        
    def build_projection(self, input_size: int, features_size: int, dropout: float) -> nn.Module:
        return Projection(input_size, features_size, dropout)

    def project(self, x: Tensor) -> Tensor:
        return self.projector(x)

    def forward(self, graphs: Data, data: Data, **kwargs) -> Tuple[Tensor, ...]:
        raise NotImplementedError
    
    def compute_loss(self, outputs: Tensor, graphs: Data, data: Data) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError


class EgoPackTask(object):
    
    # pylint: disable=unused-argument
    def align_for_egopack(self, features: Tensor, pos: Tensor, batch: Tensor, depth: Tensor, data: Data) -> Tensor:
        return features
