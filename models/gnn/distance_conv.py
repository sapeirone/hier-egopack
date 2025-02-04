import torch
import torch.nn.functional as F
from torch.nn import GELU, Parameter
from torch_geometric.nn import MessagePassing, Sequential, Linear
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import remove_self_loops, add_self_loops

import torch_geometric.nn.inits as inits

from typing import Optional, Union, List


class GatedDistanceConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 normalize: Optional[bool] = False,
                 aggr: Optional[Union[str, List[str], Aggregation]] = "add",
                 self_loops: Optional[bool] = False,
                 add_root: Optional[bool] = True,
                 activation: torch.nn.Module = torch.nn.Identity(),
                 *args, **kwargs):
        """Gated distance convolution

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        normalize : Optional[bool], optional
            L2 normalization of the output, by default False
        """
        super().__init__(aggr)

        self.normalize = normalize
        self.self_loops = self_loops
        self.add_root = add_root
        self.activation = activation

        if self.add_root:
            # projection block of the root node
            self.lin_r = Linear(in_channels, out_channels, bias=False)

        self.lin = Linear(in_channels, out_channels, bias=True)
        self.lin_dist = Sequential('x', [
            (Linear(in_channels, out_channels), 'x -> x'),
            (GELU(), 'x -> x'),
            (Linear(out_channels, out_channels), 'x -> x')
        ])
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for m in self.lin_dist.modules():
            if isinstance(m, Linear):
                m.reset_parameters()
        if self.add_root:
            self.lin_r.reset_parameters()
        inits.uniform(self.bias.shape[0], self.bias)

    def forward(self, x, edge_index):
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index)
        else:
            edge_index, _ = remove_self_loops(edge_index)

        # propagate messages and sum to root node
        out = self.propagate(edge_index, x=x) + self.bias

        # possibly add root node
        if self.add_root:
            out = out + self.lin_r(x)

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def message(self, x_i, x_j):
        dist = self.lin_dist(x_i - x_j)
        return F.sigmoid(dist) * self.activation(self.lin(x_j))
