import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)


class Projection(torch.nn.Module):

    def __init__(self, input_size: int, features_size: int = 1024, dropout: float = 0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, features_size),
            nn.LayerNorm(features_size),
            nn.ReLU(),
            nn.Linear(features_size, features_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
