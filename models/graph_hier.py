import hydra

import torch
import torch.nn as nn

from torch import Tensor

import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch

from einops.layers.torch import Rearrange

import logging

from typing import Tuple, Literal, Optional, Any

from models.conv import ConvConfig

from models.conv.conv_wrapper import GNNWrapper


logger = logging.getLogger(__name__)


class GraphHier(torch.nn.Module):
    """Hierarchical Graph Neural Network for video understanding."""
    
    def __init__(
        self,
        input_size: int,
        conv: ConvConfig,
        k: float = 2.0,
        n_layers: int = 2,
        use_norm: bool = True,
        hidden_size: int = 1024,
        depth: int = 3,
        projection_type: Literal['simple', 'mlp'] = 'simple',
        projection_dropout: float = 0,
        dropout: float = 0,
        ffn: bool = False,
        ffn_expansion_ratio: float = 1.0,
        ffn_dropout: float = 0,
        pre_norm: bool = False,
        pool: Literal['batch_subsampling', 'video_subsampling', 'max', 'mean'] = 'batch_subsampling',
        *args,
        **kwargs,
    ):
        """Build a Graph hierarchy using graph convolutional layers and fixed pooling (resolution halving).

        Parameters
        ----------
        input_size : int
            number of input features
        conv: ConvConfig
            configuration for the GNN layers
        k: float
            radius for estimating the connectivity of the input graph
        n_layers: int
            number of graph conv modules at each layer
        use_norm: bool
            use layer normalization between gnn layers
        hidden_size : int, optional
            number of hidden features, by default 1024
        depth : int, optional
            number of layers in the hierarchy, by default 3
        projection_type: Literal['simple', 'mlp']
            projection type for the first layer (simple or mlp)
        projection_dropout : float, optional
            dropout in the projection layer, by default 0.1
        dropout : float, optional
            dropout in the GNN layers, by default 0.1
        ffn: bool
            add a ffn after each gnn layer
        ffn_expansion_ratio: float
            hidden size expansion ratio for the ffn
        ffn_dropout: float
            dropout in the ffn layers
        pre_norm: bool
            use pre normalization before gnn and ffn layers
        pool: Literal['batch_subsampling', 'video_subsampling', 'max', 'mean']
            pooling method for the temporal dimension
        """
        super().__init__(*args, **kwargs)

        self.k = k
        self.hidden_size = hidden_size

        logger.info("")
        logger.info(f"Initializing GraphHier (input_size={input_size}, hidden_size={hidden_size})")

        logger.info(f"Using {projection_type} projection layer from {input_size} to {hidden_size} with dropout {projection_dropout}.")
        self.proj = self.build_projection(projection_type, input_size, hidden_size, projection_dropout)

        logger.info(f"Building {depth} layers of GNNs with hidden size {hidden_size}.")
        self.layers = nn.ModuleList([self.build_layer(conv, n_layers, dropout, use_norm) for _ in range(depth)])

        # Pre-normalization layers for the GNN and the FFN layers
        self.gnn_norm_layers = nn.ModuleList([nn.LayerNorm(hidden_size) if pre_norm else nn.Identity() for _ in range(depth)])
        self.ffn_norm_layers = nn.ModuleList([nn.LayerNorm(hidden_size) if pre_norm else nn.Identity() for _ in range(depth)])

        if ffn:
            ffn_hidden_size = int(hidden_size * ffn_expansion_ratio)
            self.ffn = nn.ModuleList([self.build_ffn_layer(hidden_size, ffn_hidden_size, ffn_dropout) for _ in range(depth)])

        self.pool = pool
        logger.info(f"Using {self.pool} pooling")

    @property
    def depth(self) -> int:
        """Return the depth of the GNN hierarchy.

        Returns
        -------
        int
            the number of layers in the GNN hierarchy
        """
        return len(self.layers)

    def build_projection(self, projection_type: Literal['simple', 'mlp'], input_size: int, hidden_size: int, dropout: float) -> nn.Module:
        """Build a projection layer for the input features

        Parameters
        ----------
        projection_type: Literal['simple', 'mlp']
            projection type for the first layer (simple or mlp)
        input_size : int
            number of input features
        hidden_size : int
            number of hidden (output) features
        dropout : float
            dropout in the projection layer

        Returns
        -------
        nn.Module
            a sequential model with the projection components
        """

        if projection_type == 'simple':
            return nn.Sequential(Rearrange("batch segments features -> batch (segments features)"), nn.Linear(input_size, hidden_size))
        else:  # 'mlp'
            return nn.Sequential(
                Rearrange("batch segments features -> batch (segments features)"), 
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            )

    def build_layer(self, conv_config: ConvConfig, n_layers: int = 2, dropout: float = 0.0, use_norm: bool = True) -> nn.Module:
        """Build a layer in the GNN hierarchy.

        Parameters
        ----------
        conv_config : Dict[Any, Any]
            configuration of the GNN layers
        dropout : float
            dropout value

        Returns
        -------
        gnn.Sequential
            a sequential model with the layer components

        Raises
        ------
        NotImplementedError
            if the normalization layer is not supported
        """
        layers = []

        layers.append((nn.Dropout(dropout), "x -> x"))

        # 1. convolutional layer
        for _ in range(n_layers - 1):
            layers.append(layer := self.build_conv_layer(conv_config))
            layers.append((nn.LeakyReLU(.2), 'x -> x'))

        layers.append(layer := self.build_conv_layer(conv_config))

        # 2. normalization layer
        if use_norm:
            layers.append((nn.LayerNorm(conv_config['out_channels']), 'x -> x'))  # type: ignore

        # 3. activation layer
        layers.append((nn.LeakyReLU(.2), 'x -> x'))

        return gnn.Sequential("x, edge_index, pos, batch", layers)

    def build_conv_layer(self, config: Any) -> Tuple[nn.Module, str]:
        conv_layer = hydra.utils.instantiate(config)

        if isinstance(conv_layer, GNNWrapper):
            return (conv_layer, "x, edge_index, pos -> x")

        return (conv_layer, "x, edge_index -> x")

    def build_ffn_layer(self, input_features, hidden_size, dropout: float = 0):
        return nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_features),
            nn.Dropout(dropout)
        )

    def forward(self, data: Data, max_depth: Optional[int] = None, *args, **kwargs) -> Tuple[torch.Tensor, Batch]:
        """Forward the graphs through the hierarchical layers of the GNN.

        Parameters
        ----------
        data : Data
            input graphs

        Returns
        -------
        Tuple[torch.Tensor, BaseData]
            features of the last layer and intermediate graphs
        """

        # data may be padded to a fixed temporal dimension
        # we can safely ignore the padding when processing the inputs as graphs 
        # (that would not be the case if we used conv1d layers)
        x, batch, pos, indices = data.x, data.batch, data.pos, data.indices
        x, batch, pos, indices = x[data.mask], batch[data.mask], pos[data.mask], indices[data.mask]  # type: ignore

        # compute the initial adjacency matrix of the graph
        edge_index = gnn.radius_graph(pos, self.k, batch, False)

        feat = self.proj(x)

        # intermediate graphs, after each layer of the hierarchical GNN
        int_graphs = []

        # GNN + Downsampling
        # for depth, (layer_l, layer_r) in enumerate(zip(self.layers_l, self.layers_r)):
        for depth, layer in enumerate(self.layers):
            if max_depth is not None and depth > max_depth:
                break

            gnn_norm_layer = self.gnn_norm_layers[depth]

            feat = feat + layer(gnn_norm_layer(feat), edge_index, pos, batch)

            if hasattr(self, 'ffn'):
                ffn_layer: nn.Module = self.ffn[depth]
                ffn_norm_layer = self.ffn_norm_layers[depth]
                feat = feat + ffn_layer(ffn_norm_layer(feat))

            int_graphs.append(
                Data(
                    x=feat,
                    pos=pos,
                    video=batch,
                    depth=torch.ones_like(pos, dtype=torch.long) * depth,
                    # TODO: redundant to have both depth and extent
                    extent=torch.ones_like(pos, dtype=torch.float) * (2 ** depth),
                    indices=indices,
                )
            )

            # Apply temporal pooling to the graph at this layer
            feat, pos, batch, indices = self.time_pooling(feat, pos, batch, indices)

            edge_index = gnn.radius_graph(pos / (2.0 ** (depth + 1)), self.k, batch, False)

        int_graphs = Batch.from_data_list(int_graphs, follow_batch=["video"])

        return feat, int_graphs

    def time_pooling(self, x: Tensor, pos: Tensor, batch: Tensor, indices: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Perform time pooling on the input features.

        Parameters
        ----------
        x : Tensor
            input features
        pos : Tensor
            input positions
        batch : Tensor
            input batch
        indices : Tensor
            input indices

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor]
            temporally pooled features, positions, batches, and indices
        """

        if self.pool == 'batch_subsampling':
            # Return one sample every two, regardless of video boundaries
            return x[::2], pos[::2], batch[::2], indices[::2]

        if self.pool == 'video_subsampling':
            mask = (indices % 2 == 0)

            return x[mask], pos[mask], batch[mask], indices[mask] // 2

        if self.pool in ['max', 'mean']:
            pooling_edges = gnn.radius_graph(indices.float(), 1.5, batch, True)

            # create a data object on the fly 
            data = Data(x=x, edge_index=pooling_edges)
            data = gnn.pool.max_pool_neighbor_x(data) if self.pool == 'max' else gnn.pool.avg_pool_neighbor_x(data)

            x = data.x  # type: ignore

            mask = (indices % 2 == 0)

            return x[mask], pos[mask], batch[mask], indices[mask] // 2

        raise NotImplementedError(f"Pooling method {self.pool} not implemented.")