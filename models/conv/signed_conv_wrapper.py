import torch
from torch import Tensor

from torch_geometric.nn.conv import SignedConv

from models.conv.conv_wrapper import GNNWrapper


class SignedDistanceConvWrapper(SignedConv, GNNWrapper):
    r"""Wrapper for SignedConv."""

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        """Initialize the Temporal Distance Gated Convolution layer.

        Parameters
        ----------
        in_channels : int
            size of input features
        out_channels : int
            size of output features
        """
        super(SignedDistanceConvWrapper, self).__init__(in_channels, out_channels // 2, *args, **kwargs)

    def forward(self, x: Tensor, edge_index: torch.Tensor, pos: torch.Tensor) -> Tensor:

        sign = torch.sign(pos[edge_index[0]] - pos[edge_index[1]])
        return super(SignedDistanceConvWrapper, self).forward(x, edge_index[:, sign > 0], edge_index[:, sign < 0])
