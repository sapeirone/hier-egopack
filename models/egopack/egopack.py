"""
EgoPack module.
"""

from typing import Literal, Dict, List, Tuple

import logging

import torch
from torch import Tensor

import torch.nn as nn
import torch_geometric.nn as gnn

logger = logging.getLogger(__name__)


class EgoPack(nn.Module):
    """EgoPack interaction module."""

    def __init__(self,
                 task_prototypes: Dict[str, Tensor],
                 features_size: int = 256,
                 hidden_size: int = 256,
                 # interaction
                 k: int = 8,
                 depth: int = 3,
                 distance_func: Literal["l2", "cosine"] = "cosine",
                 # GNN parameters 
                 conv: Literal["sage", "graph"] = "graph",
                 conv_depth: int = 2,
                 dropout: float = 0,
                 *args, **kwargs) -> None:
        """Initialize the EgoPack module.

        Parameters
        ----------
        task_prototypes : Dict[str, Tensor]
            task-specific prototypes
        features_size : int, optional
            input features size, by default 256
        hidden_size : int, optional
            hidden size in the task interaction process, by default 256
        k : int, optional
            number of task prototypes to match, by default 8
        depth : int, optional
            depth of the interaction stages, by default 3
        distance_func : Literal["l2", "cosine"], optional
            distance function used in the matching process, by default "cosine"
        conv : Literal["sage", "graph"], optional
            interaction GNN, by default "graph"
        conv_depth : int, optional
            depth of the interaction GNN, by default 2
        dropout : float, optional
            dropout in the interaction layers, by default 0
        """
        super().__init__()

        self.feature_size = features_size

        # Initialize the task prototypes
        self.task_prototypes = nn.ModuleDict({task: nn.Embedding.from_pretrained(prototypes, freeze=True) for task, prototypes in task_prototypes.items()})
        
        logger.info("Initializing EgoPack with %d tasks using depth = %d and K = %d.", len(task_prototypes), depth, k)
        logger.info("Using %s conv layers, conv_depth = %d and hidden_size = %d.", conv, conv_depth, hidden_size)

        # Interaction parameters
        self.k = k
        self.distance_func = distance_func
                    
        # Interaction stages (one module for each task)
        self.stages = nn.ModuleDict({
            task: nn.ModuleList([self.build_interaction_layer(features_size, hidden_size, dropout, conv, conv_depth) for _ in range(depth)])
            for task in self.task_labels
        })
        
    @property
    def task_labels(self) -> List[str]:
        """Get the list of auxiliary tasks.

        Returns
        -------
        List[str]
            list of auxiliary tasks
        """
        return list(self.task_prototypes.keys())
        
    def build_interaction_layer(self, input_size: int, hidden_size: int, dropout: float = 0, 
                                conv: Literal['sage', 'graph'] = 'graph', depth: int = 2) -> gnn.Sequential:
        """Build an interaction stage for the EgoPack process.

        Parameters
        ----------
        input_size : int
            size of the input features
        hidden_size : int
            hidden size
        dropout : float, optional
            dropout, by default 0
        conv : Literal["sage", "graph"], optional
            gnn, by default 'graph'
        depth : int, optional
            number of gnn layers in each stage, by default 2

        Returns
        -------
        gnn.Sequential
            EgoPack interaction stage
        """
        
        layers = []
        
        layers.append((nn.Dropout(dropout), 'x -> x'))
        
        for i in range(depth):
            if conv == 'sage':
                layers.append((gnn.SAGEConv(input_size if i == 0 else hidden_size, hidden_size, root_weight=True), 'x, edge_index -> x'))
            else:
                layers.append((gnn.GraphConv(input_size if i == 0 else hidden_size, hidden_size, aggr='sum', bias=True), 'x, edge_index, weights -> x'))
                
            if i < depth - 1:
                # only apply activation function between intermediate layers
                layers.append((nn.LeakyReLU(0.2), 'x -> x'))
        
        # Map back to the input size
        layers.append((nn.Linear(hidden_size, input_size), 'x -> x'))
        
        return gnn.Sequential('x, edge_index, weights', layers)

    def interact(self, task_features: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, Dict[int, Tensor]]]:
        """EgoPack interaction.

        Parameters
        ----------
        task_features : Dict[str, Tensor]
            task-specific features

        Returns
        -------
        Tuple[Dict[str, Tensor], Dict[str, Dict[int, Tensor]]]
            task-features after interaction and matches
        """
        output_features: Dict[str, Tensor] = dict()
        matches: Dict[str, Dict[int, Tensor]] = dict()

        for task, features in task_features.items():
            output_features[task], matches[task] = self.task_interaction(task, features, self.task_prototypes[task].weight)

        return output_features, matches
    
    def forward(self, task_features: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, Dict[int, Tensor]]]:
        return self.interact(task_features)

    def task_interaction(self, task: str, features: Tensor, prototypes: Tensor) -> Tuple[Tensor, Dict[int, Tensor]]:
        conv_stages: nn.ModuleList = self.stages[task]  # type: ignore

        # Keep track of the matched nodes at each depth
        matches = dict()

        edges = None
        for depth, conv in enumerate(conv_stages):

            # Compute edges for graph - prototypes interaction
            edges, matches[depth], weights = self._compute_edges(features, prototypes)
            
            graph = torch.cat([prototypes, features], dim=0)
            graph = conv(graph, edges, weights)
            
            features = graph[-features.shape[0]:]

        return features, matches

    @torch.no_grad()
    def _compute_edges(self, features: Tensor, prototypes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute edges for graph - prototypes interaction.

        Parameters
        ----------
        features : Tensor
            task features
        prototypes : Tensor
            task specific prototypes

        Returns
        -------
        Tuple[Tensor, Tensor]
            computed edges and closest node to each input feature
        """

        num_prototypes = prototypes.shape[0]
        batch_size = features.shape[0]

        # Get closest prototypes for each node in features
        if self.distance_func == 'l2':
            # l2 distance (lower is better)
            distances = cdist(features, prototypes)
        else:
            # cosine distance by default
            distances = cos_dissimilarity(features, prototypes)

        # Sort nodes by distance in ascending order
        matches = distances.argsort(dim=-1, descending=False)[:, :self.k]
        distances = distances.sort(dim=-1, descending=False).values[:, :self.k]
        
        # Compute edges going out from the closest nodes and entering the features
        # Given that CN{1..8} identify the closest nodes to feature 1, the edges are:
        # [CN1, 0], [CN2, 0], [CN3, 0], [CN4, 0], [CN5, 0], [CN6, 0], [CN7, 0], [CN8, 0], ...
        edges = torch.stack([
            matches.flatten(), 
            torch.arange(num_prototypes, num_prototypes + batch_size, device=prototypes.device).repeat_interleave(matches.shape[1])
        ])
        assert edges.shape == (2, batch_size * self.k)
        
        weights = (1 - distances.flatten()).relu()
        
        # Return the edges and the closest node to each input feature
        return edges, matches[:, 0], weights


def cdist(g1, g2):
    return torch.cdist(g1, g2, p=2, compute_mode='donot_use_mm_for_euclid_dist')


def cos_dissimilarity(g1, g2):
    g1 = g1 / g1.norm(dim=1, keepdim=True)
    g2 = g2 / g2.norm(dim=1, keepdim=True)
    return (1 - torch.mm(g1, g2.T))


def cossim(g1, g2):
    g1 = g1 / g1.norm(dim=1, keepdim=True)
    g2 = g2 / g2.norm(dim=1, keepdim=True)
    return torch.mm(g1, g2.T)
