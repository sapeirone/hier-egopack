import torch
from torch import Tensor

from torch_geometric.nn.conv import SAGEConv

from torch_geometric.nn.encoding import PositionalEncoding


from models.conv.conv_wrapper import GNNWrapper


class SAGEWithPEWrapper(SAGEConv, GNNWrapper):
    r"""Wrapper for SAGE with PE."""

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        """Initialize the SAGE with PE Convolution layer.

        Parameters
        ----------
        in_channels : int
            size of input features
        out_channels : int
            size of output features
        """
        super(SAGEWithPEWrapper, self).__init__(in_channels, out_channels, *args, **kwargs)
        self.pe = PositionalEncoding(in_channels)

    def forward(self, x: Tensor, edge_index: torch.Tensor, pos: torch.Tensor) -> Tensor:
        return super(SAGEWithPEWrapper, self).forward(x + self.pe(pos), edge_index)
