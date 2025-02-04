import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from models.conv.conv_wrapper import GNNWrapper


from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Size, OptTensor

from typing import Literal


class DistanceGatedConv(MessagePassing, GNNWrapper):
    r"""Temporal Distance Gated Convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        aggr: str = 'mean',
        pos_hidden_channels: int = 256,
        pos_act: Literal['sigmoid', 'tanh', 'relu'] = 'tanh',
        use_signed_temporal_distance: bool = True,
        use_pos_attr: bool = True,
        **kwargs,
    ):
        """Initialize the Temporal Distance Gated Convolution layer.

        Parameters
        ----------
        in_channels : int
            size of input features
        out_channels : int
            size of output features
        bias : bool, optional
            additional bias term, by default True
        aggr : str, optional
            aggregation function, by default 'mean'
        pos_hidden_channels : int, optional
            number of hidden channels in the learnable position projection, by default 256
        pos_act : Literal["sigmoid", "tanh", "relu"], optional
            activation function of the learnable position projection, by default 'tanh'
        use_signed_temporal_distance : bool, optional
            use signed temporal distance as edge weight in the aggregation phase, by default True
        use_pos_attr : bool, optional
            use the absolute relative distance between nodes as edge attribute, by default True
        """
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_rel = Linear(in_channels, out_channels, bias=True)
        self.lin_root = Linear(in_channels, out_channels, bias=False)
        
        activations = {'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh(), 'relu': nn.ReLU()}
        self.lin_pos = nn.Sequential(
            nn.Linear(1, pos_hidden_channels), 
            nn.LeakyReLU(), 
            nn.Linear(pos_hidden_channels, out_channels), 
            activations[pos_act]
        )
        
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(1, out_channels))
            
        self.use_signed_temporal_distance = use_signed_temporal_distance
        self.use_pos_attr = use_pos_attr

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # self.lin_rel.reset_parameters()
        torch.nn.init.eye_(self.lin_rel.weight)
        torch.nn.init.zeros_(self.lin_rel.bias)
        self.lin_root.reset_parameters()
        if hasattr(self, 'bias'):
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, 
                edge_index: torch.Tensor, 
                pos: torch.Tensor, 
                size: Size = None) -> Tensor:

        root = x

        # Perform Message Passing separately for left and right nodes
        edge_weight = None
        if self.use_signed_temporal_distance:
            edge_weight = torch.sign(pos[edge_index[0]] - pos[edge_index[1]])
        else:
            edge_weight = torch.ones_like(pos[edge_index[0]] - pos[edge_index[1]])
        
        edge_attr = None
        if self.use_pos_attr:
            edge_attr = self.lin_pos((pos[edge_index[0]] - pos[edge_index[1]]).unsqueeze(-1).abs())
        
        # Project the neighours
        x = self.lin_rel(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        # Aggregation step
        out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr, edge_weight=edge_weight, size=size)
        out = out + self.lin_root(root)
        
        if hasattr(self, 'bias'):
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_attr: OptTensor, edge_weight: OptTensor) -> Tensor:
        out = x_j
        
        if edge_attr is not None:
            out = out * edge_attr
        
        if edge_weight is not None:
            out = out * edge_weight.unsqueeze(1)
        
        return out
