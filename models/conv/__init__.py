from dataclasses import dataclass
from typing import Union


@dataclass
class SAGE:
    project: bool
    normalize: bool
    root_weight: bool
    aggr: str
    

class DGC:
    bias: bool
    pos_hidden_channels: int
    aggr: str


@dataclass
class Linear:
    pass
    
# add additional configurations for other convolutional layers here
ConvConfig = Union[SAGE, DGC, Linear] 