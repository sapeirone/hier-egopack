"""Ego4d Object State Change Classification Task"""

import logging
from typing import Tuple, Literal, Optional, List, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Data

from models.tasks.task import Task, EgoPackTask
from models.tasks.utils.classifier import Classifier


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OSCCTask(Task):
    """Ego4D Object State Change Classification (OSCC) task."""

    def __init__(self, 
                 name: str,
                 input_size: int, 
                 features_size: int, 
                 dropout: float = 0, 
                 head_dropout: float = 0,
                 loss_weight: float = 1.0,
                 **kwargs
        ):
        """Initialize the Ego4D Object State Change Classification (OSCC) task.

        Parameters
        ----------
        name : str
            task name
        input_size : int
            size of the input features
        features_size : int
            size of the projected features
        dropout : float, optional
            dropout in the projection layer, by default 0
        head_dropout : float, optional
            dropout in the classification heads, by default 0
        loss_weight : float, optional
            weight of the classification loss, by default 1.0
        """
        
        logger.info("Initializing %s task with input size %d and features_size %d.", name, input_size, features_size)
        
        super().__init__(name, input_size, features_size, dropout)
        
        self.classifier: torch.nn.Module = Classifier(features_size, 2, dropout=head_dropout, prior_prob=0)

        # Loss configuration
        self.loss_weight = loss_weight
        
    def align(self, features: Tensor, pos: Tensor, batch: Tensor, segments: Tensor, segments_batch: Tensor) -> Tensor:
        """Align node positions with the ground truth segments and pool features.
        
        We use start and end boundaries of the action segments (according to the ground truth)
        to max (or mean) pool features for each action.

        Parameters
        ----------
        features : Tensor
            input features (tensor shape: [batch_size, input_size])
        pos : Optional[Tensor], optional
            node positions
        batch : Tensor
            batch tensor
        segments : Tensor
            ground truth start and end boundaries of the actions (tensor shape: [batch_size, 2])
        segments_batch : Tensor
            batch tensor for the segments

        Returns
        -------
        Tensor
            pooled features
        """
        
        # pos.shape = (N), batch.shape = (N), segments.shape = (M, 2), segments_batch.shape = (M)
        feat_size = features.shape[1]
        
        mask = segments_batch[:, None] == batch[None, :]  # node and gt are from the same video
        mask = torch.logical_and(mask, segments[:, :1] <= pos[None, :])
        mask = torch.logical_and(mask, segments[:, 1:] >= pos[None, :])
        
        count = mask.sum(1)  # number of nodes for segment
        
        # for each row i in mask, torch.argwhere(mask[i]) returns the indices that should 
        # be pooled for the corresponding ground truth
        
        mask = mask[:, :, None].expand(-1, -1, feat_size)
        features = mask * features[None]

        return features.sum(1) / count[:, None]
            
    def forward(self, graphs: Data, data: Data, **kwargs) -> Tuple[Tensor, ...]:
        """Forward features through the projection module and the classifiers.

        Parameters
        ----------
        graphs : Data
            output of the temporal backbone
        data : Data
            input graphs (from the dataset)

        Returns
        -------
        Tensor
            output logits (tensor shape: [batch_size, n_classes]) and aligned features
        """
        features, pos, batch, depth, segments, segments_batch = graphs.x, graphs.pos, graphs.video, graphs.depth, data.segments, data.segments_batch
        
        # Make sure to keep only features from high-resolution features for this task
        features, pos, batch = features[depth == 0], pos[depth == 0], batch[depth == 0]  # type: ignore
        
        # Step 1: features projection
        features = self.project(features)
        
        # Step 2: align node positions with the ground truth segments and pool features.
        # We use start and end boundaries of the action segments (according to the ground truth)
        # to max (or mean) pool features for each action.
        features = self.align(features, pos, batch, segments, segments_batch)
        
        # Step 3: classification
        return self.classifier(features), features
    
    def compute_loss(self, outputs: Tensor, graphs: Data, data: Data) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """Compute the loss function for the PNR task.
        
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
        logits, _ = outputs
        loss = F.cross_entropy(logits, data.labels, ignore_index=-1, reduction="none")
        return self.loss_weight * loss, (loss, )


class EgoPackOSCCTask(OSCCTask, EgoPackTask):
    """Ego4D EgoPack Object State Change Classification (OSCC) task."""
    
    def __init__(self,
                 features_size: int, 
                 head_dropout: float = 0.0,
                 # EgoPack-specific configuration
                 fusion_level: Literal['features', 'logits', 'none'] = 'features',
                 fusion_dropout: float = 0.1,
                 aux_tasks: Optional[List[str]] = None,
                 **kwargs
        ):
        super().__init__(features_size=features_size, head_dropout=head_dropout, **kwargs)

        self.fusion_level: Literal['features', 'logits', 'none'] = fusion_level
        
        if fusion_level == 'logits':
            assert aux_tasks is not None, 'Auxiliary tasks not provided'
            self.aux_classifier = nn.ModuleDict({task: Classifier(features_size, 2, dropout=head_dropout, prior_prob=0, bias=False) for task in aux_tasks})
        elif fusion_level == 'features':
            self.fusion_dropout: float = fusion_dropout
            
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
        # Make sure to keep only features from high-resolution features for this task
        features, pos, batch = features[depth == 0], pos[depth == 0], batch[depth == 0]  # type: ignore
        
        return self.align(features, pos, batch, data.segments, data.segments_batch)
            
    def forward(self, graphs: Data, data: Data, auxiliary_features: Dict[str, Tensor], **kwargs) -> Tuple[Tensor, ...]:
        """Forward features through the projection module and the classifiers.

        Parameters
        ----------
        graphs : Data
            output of the temporal backbone
        data : Data
            input graphs (from the dataset)
        auxiliary_features: Dict[str, Tensor]
            auxiliary features from secondary tasks

        Returns
        -------
        Tensor
            output logits (tensor shape: [batch_size, n_classes])
        """
        features, pos, batch, depth, segments, segments_batch = graphs.x, graphs.pos, graphs.video, graphs.depth, data.segments, data.segments_batch
        
        # Make sure to keep only features from high-resolution features for this task
        features, pos, batch = features[depth == 0], pos[depth == 0], batch[depth == 0]  # type: ignore
    
        # Step 1+2: Project and align main task features
        features = self.project(features)
        features = self.align(features, pos, batch, segments, segments_batch)
        
        # Step 3: Features-level fusion
        if self.fusion_level == 'features':
            # Features-level fusion: sum the main task features and the auxiliary features
            features = features + torch.stack(list(auxiliary_features.values())).mean(0)
        
        # Step 4: Forward the features of the main task through the OSCC classifier
        logits: Tensor = self.classifier(features)

        # Step 5: Logits-level fusion
        if self.fusion_level == 'logits':
            # Forward the auxiliary features through their specific classifiers
            aux_logits = [self.aux_classifier[task](f) for task, f in auxiliary_features.items()]
            logits: Tensor = torch.stack([logits, *aux_logits]).mean(0)
        
        return logits, features
