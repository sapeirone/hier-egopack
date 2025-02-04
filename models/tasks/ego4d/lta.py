"""Ego4d LTA Task"""

import logging
from typing import Tuple, Dict, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from torch_geometric.data import Data
from torch_geometric import nn as gnn

from models.tasks.task import Task, EgoPackTask
from models.tasks.utils.classifier import Classifier
from models.conv.distance_gated_conv import DistanceGatedConv as DGC


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class LTATask(Task):
    """Ego4d Long-Term Anticipation (LTA) task."""

    def __init__(self,
                 name = 'ego4d/lta',
                 input_size: int = 768,
                 features_size: int = 768,
                 K: int = 5,
                 num_verbs: int = 115,
                 num_nouns: int = 478,
                 dropout: float = 0,
                 head_dropout: float = 0,
                 loss_weight: float = 1.0,
                 **kwargs  # pylint: disable=unused-argument
        ):
        """Initialize the LTA Task.
        
        As other Ego4D Tasks, the LTA head is implemented as a neck + head architecture.
        The neck is implemented as a MLP, while the head performs two layers of graph
        convolution using TDGC layers: input_graph -> TDGC -> ReLU -> TDGC -> ReLU.

        Parameters
        ----------
        name : str, optional
            task name, by default 'ego4d/lta'
        input_size : int, optional
            size of the input features, by default 768
        features_size : int, optional
            size of the projected features, by default 768
        K : int, optional
            number of future actions to predict, by default 20
        num_verbs : int, optional
            number of verb labels in the dataset, by default 115
        num_nouns : int, optional
            number of noun labels in the dataset, by default 478
        dropout : float, optional
            dropout in the projection neck, by default 0
        head_dropout : float, optional
            dropout in the classification heads, by default 0
        loss_weight : float, optional
            weight of the classification loss, by default 1.0
        """
        
        logger.info("Initializing %s task with input size %d and features_size %d.", name, input_size, features_size)
        
        super().__init__(name, input_size, features_size, dropout)
        
        self.k = K
        
        # LTA TDGC-based head
        self.head = self.build_lta_head(features_size)

        # Verbs and Nouns classifiers
        self.verb_classifier: torch.nn.Module = Classifier(features_size, num_verbs, dropout=head_dropout)
        self.noun_classifier: torch.nn.Module = Classifier(features_size, num_nouns, dropout=head_dropout)

        self.loss_weight = loss_weight
        
    def build_lta_head(self, features_size: int) -> nn.Module:
        """Build LTA TDGC-based head.

        Parameters
        ----------
        features_size : int
            size of the input features

        Returns
        -------
        nn.Module
            the LTA head module
        """
        return gnn.Sequential('x, edge_index, pos', [
            (DGC(features_size, features_size, aggr='mean'), 'x, edge_index, pos -> x'),
            (nn.ReLU(), 'x -> x'),
            (DGC(features_size, features_size, aggr='mean'), 'x, edge_index, pos -> x'),
            (nn.ReLU(), 'x -> x'),
        ])
        
    def align(self, features: Tensor, batch: Tensor) -> Tensor:
        """Align node positions with the ground truth segments and pool features.
        
        Practically, this method applies global max pooling to each video in the batch separately.

        Parameters
        ----------
        features : Tensor
            input features (tensor shape: [batch_size, input_size])
        batch : Tensor
            batch tensor (tensor shape: [batch_size])

        Returns
        -------
        Tensor
            video pooled features
        """
        return gnn.pool.global_max_pool(features, batch)
    
    def build_lta_graph(self, pooled_features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Build the LTA graph for the video features.

        To build the LTA features are repeated K times and the edges are created using
        a radius graph with fixed radius.

        Parameters
        ----------
        pooled_features : Tensor
            video-level features

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            features, edges and position tensors of the LTA graph
        """
        num_videos = pooled_features.shape[0]
        
        # Build the nodes of the LTA graph by repeating the pooled features k times
        features = torch.repeat_interleave(pooled_features, self.k, dim=0)
        # Define the batch tensor for the graph
        batch = torch.arange(num_videos, device=features.device).repeat(self.k)  # type: ignore
        # Define the position of the nodes (0...K indices)
        pos = torch.arange(self.k, device=features.device, dtype=float).repeat(num_videos).float()  # type: ignore
        # Fixed size local connectivity
        edge_index = gnn.radius_graph(pos, 2.5, batch, False)
        
        return features, edge_index, pos

    def forward(self, graphs: Data, data: Data, **kwargs) -> Tuple[Tensor, ...]:
        """Forward features through the projection module and the classifiers.

        Parameters
        ----------
        features : Tensor
            input features (tensor shape: [batch_size, input_size])
        batch : Optional[Tensor], optional
            batch tensor, by default None

        Returns
        -------
        Tuple[Tensor, ...]
            output logits (tensor shape: [batch_size, n_classes])
        """
        features, pos, batch, depth = graphs.x, graphs.pos, graphs.video, graphs.depth
        
        # Make sure to keep only features from high-resolution features for this task
        features, pos, batch = features[depth == 0], pos[depth == 0], batch[depth == 0]  # type: ignore
        
        # Step 1: project and pool video features
        features = self.project(features)  # features have shape [num_nodes, features_size]
        video_features = self.align(features, batch)  # features have shape [num_unique_videos, features_size]
        
        # Step 2: build online the LTA graph and apply the head
        lta_features, lta_edge_index, lta_pos = self.build_lta_graph(video_features)
        lta_features = self.head(lta_features, lta_edge_index, lta_pos)
        
        # Step 3: verb and noun classifier
        verb_logits, noun_logits = self.verb_classifier(lta_features), self.noun_classifier(lta_features)
        return verb_logits, noun_logits, lta_features

    def compute_loss(self, outputs: Tuple[Tensor, ...], graphs: Data, data: Data) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Compute the loss function for the LTA task.
        
        See docs from the Task class for more details on the input arguments.

        Parameters
        ----------
        outputs : Tuple[Tensor, ...]
            task outputs
        graphs : Data
            temporal backbone outputs
        data : Data
            input data

        Returns
        -------
        Tuple[Tensor, Tuple[Tensor, Tensor]]
            the total loss, scaled by the loss_weight of the task, as well as 
            the individual components of the loss
        """
        verb_logits, noun_logits, _ = outputs
        verb_labels, noun_labels = data.labels
        
        # Compute the loss for the verb and noun logits
        verb_loss = F.cross_entropy(verb_logits, verb_labels, ignore_index=-1, reduction="none")
        noun_loss = F.cross_entropy(noun_logits, noun_labels, ignore_index=-1, reduction="none")

        total_loss = self.loss_weight * (verb_loss + noun_loss)
        return total_loss, (verb_loss, noun_loss)
    

class EgoPackLTATask(LTATask, EgoPackTask):
    """EgoPack Task for LTA"""
    
    def __init__(self, *args, **kwargs):
        """EgoPack LTA Task.

        Initialize the EgoPack LTA Task.

        Parameters
        ----------
        fusion_level : Literal['features', 'logits', 'none'], optional
            fusion level for egopack interaction, by default 'features'

        Raises
        ------
        ValueError
            if fusion_level is not 'features' 
        """
        super().__init__(*args, **kwargs)
        
        # This task only support features level interaction for EgoPack.
        # For the moment we keep this argument for compatibility with other
        # tasks but we could consider dropping it in the future.
            
    def align_for_egopack(self, features: Tensor, pos: Tensor, batch: Tensor, depth: Tensor, data: Data) -> Tensor:
        """Align auxiliary features to this task.

        Parameters
        ----------
        features : Tensor
            auxiliary features
        pos : Tensor
            position tensor for the input features
        batch : Tensor
            batch tensor for the input features

        Returns
        -------
        Tensor
            task-aligned features
        """
        # Make sure to keep only features from high-resolution features
        features, pos, batch = features[depth == 0], pos[depth == 0], batch[depth == 0]  # type: ignore
        
        return self.align(features, batch)
            
    # pylint: disable=unused-argument,arguments-differ
    def forward(self, graphs: Data, data: Data, aux_features: Dict[str, Tensor], **kwargs) -> Tuple[Tensor, ...]:
        """Forward features through the projection module and the classifiers.

        Parameters
        ----------
        graphs : Data
            output of the temporal GNN
        data : Optional[Tensor], optional
            input data to the model (may be used to retrieve labels or other metadata on the input)
        auxiliary_features : Optional[Dict[str, Tensor]], optional
            auxiliary features from other tasks

        Returns
        -------
        Tensor
            output logits (tensor shape: [batch_size, n_classes])
        """
        features, pos, batch, depth = graphs.x, graphs.pos, graphs.video, graphs.depth
        
        # Make sure to keep only features from high-resolution features
        features, pos, batch = features[depth == 0], pos[depth == 0], batch[depth == 0]  # type: ignore
        
        # Step 1: project and pool video features
        features = self.project(features)  # features have shape [num_nodes, features_size]
        video_features = self.align(features, batch)  # features have shape [num_unique_videos, features_size]
        
        # Merge the main task features with those from the auxiliary tasks
        video_features = video_features + torch.stack(list(aux_features.values())).mean(0)
        
        # Step 3: build online the LTA graph and apply the head
        lta_features, lta_edge_index, lta_pos = self.build_lta_graph(video_features)
        lta_features = self.head(lta_features, lta_edge_index, lta_pos)
        
        # Step 4: verb and noun classifier
        verb_logits, noun_logits = self.verb_classifier(lta_features), self.noun_classifier(lta_features)
        return verb_logits, noun_logits, lta_features


if __name__ == '__main__':
    pass
