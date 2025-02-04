"""Ego4d Point of No Return Task"""

import logging
from typing import Tuple, Literal, Optional, List, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn.encoding import PositionalEncoding

from models.tasks.task import Task, EgoPackTask


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PNRTask(Task):
    """Ego4D Point of No Return (PNR) task."""

    def __init__(self, 
                 name: str,
                 input_size: int, 
                 features_size: int, 
                 dropout: float = 0,
                 loss_weight: float = 1.0,
                 **kwargs
        ):
        """Initialize the PNR task.

        Parameters
        ----------
        name : str
            name of the task
        input_size : int
            size of the input features
        features_size : int
            size of the projected features
        dropout : float, optional
            dropout, by default 0
        loss_weight : float, optional
            weight of the classification loss, by default 1.0
        """
        logger.info("Initializing %s task with input size %d and features_size %d.", name, input_size, features_size)
        
        super().__init__(name, input_size, features_size, dropout, **kwargs)
        
        self.classifier: torch.nn.Module = torch.nn.Linear(features_size, 1)
        self.pe = PositionalEncoding(features_size, granularity=1)

        self.loss_weight = loss_weight
    
    # pylint: disable=unused-argument,arguments-differ
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
            output logits (tensor shape: [batch_size, n_classes])
        """        
        features, depth = graphs.x, graphs.depth
        
        # Make sure to keep only features from high-resolution features for this task
        features = features[depth == 0]  # type: ignore
        
        # Step 1: Add a positional encoding to the input features...
        features = features + self.pe(data.indices)
        # ...and project them (no alignment required for this task)
        features = self.project(features)

        # Step 2: Compute output predictions
        return (self.classifier(features), )

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
        logits, = outputs
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(), data.labels, reduction='mean')
        return self.loss_weight * loss, (loss, )
    

class EgoPackPNRTask(PNRTask, EgoPackTask):
    """EgoPack Task for PNR"""
    
    def __init__(self, features_size: int,
                 # EgoPack-specific configurations
                 fusion_level: Literal['features', 'logits'] = 'features', 
                 fusion_dropout: float = 0.1, 
                 aux_tasks: Optional[List[str]] = None,
                 **kwargs
        ):
        """Initialize the EgoPack PNR Task.

        Parameters
        ----------
        features_size : int
            features size
        fusion_level : Literal['features', 'logits'], optional
            fusion level for EgoPack, by default 'logits'
        fusion_dropout : float, optional
            dropout in the EgoPack fusion step, by default 0.1
        aux_tasks : Optional[List[str]], optional
            list of auxiliary tasks, by default None
        """
        super().__init__(features_size=features_size, **kwargs)

        self.fusion_level: Literal['features', 'logits'] = fusion_level
        
        if fusion_level == 'logits':
            assert aux_tasks is not None, 'Auxiliary tasks not provided'
            self.aux_classifier = nn.ModuleDict({task: nn.Linear(features_size, 1) for task in aux_tasks})
            
        elif fusion_level == 'features':
            self.fusion_dropout: float = fusion_dropout
            
    def align_for_egopack(self, features: Tensor, pos: Tensor, batch: Tensor, depth: Tensor, data: Data) -> Tensor:
        """Align auxiliary features to this task.

        Parameters
        ----------
        features : Tensor
            auxiliary features

        Returns
        -------
        Tensor
            task-aligned features
        """
        return features
    
    # pylint: disable=unused-argument,arguments-differ
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
        features, depth = graphs.x, graphs.depth
        
        # Make sure to keep only features from high-resolution features for this task
        features = features[depth == 0]  # type: ignore
        
        # Step 1: Add a positional encoding to the input features...
        features = features + self.pe(data.indices)
        features = self.project(features)
        
        # Step 2 (mutually exclusive with step 4): Features fusion
        if self.fusion_level == 'features':
            features = features + 0.01 * torch.stack(list(auxiliary_features.values())).mean(0)
        
        # Step 3: Main task classification
        logits: Tensor = self.classifier(features)

        # Step 4 (mutually exclusive with step 2): Logits fusion
        if self.fusion_level == 'logits':
            logits: Tensor = torch.stack([logits] + [self.aux_classifier[task](f) for task, f in auxiliary_features.items()]).mean(0)
        
        return (logits, )
