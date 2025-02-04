"""Ego4d Action Recognition Task"""

import logging
from typing import Tuple, Optional, Dict, Literal, List, Any

import hydra

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from models.tasks.task import Task, EgoPackTask

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ARTask(Task):
    """Ego4D Action Recognition task."""

    def __init__(self, 
                 name: str,
                 input_size: int, 
                 features_size: int,
                 verb_classifier: Dict[str, Any],
                 noun_classifier: Dict[str, Any],
                 dropout: float = 0,
                 loss_weight: float = 1.0,
                 **kwargs
        ):
        """Initialize the Ego4D Action Recognition task.

        Parameters
        ----------
        name : str
            task name
        input_size : int
            size of the input features
        features_size : int
            size of the projected features
        verb_classifier: Dict[str, Any]
            configuration for the verb classifier
        noun_classifier: Dict[str, Any]
            configuration for the noun classifier
        dropout : float, optional
            dropout in the projection layer, by default 0
        loss_weight : float, optional
            weight of the classification loss, by default 1.0
        """
        
        logger.info("Initializing %s task with input size %d and features_size %d.", name, input_size, features_size)

        super().__init__(name, input_size, features_size, dropout, **kwargs)
        
        # Verbs and Nouns classifiers, dynamically instantiated using hydra
        self.verb_classifier: nn.Module = hydra.utils.instantiate(verb_classifier)
        self.noun_classifier: nn.Module = hydra.utils.instantiate(noun_classifier)

        # Loss configuration
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        self.loss_weight = loss_weight
        
    def align(self, graphs: Data, data: Data, *args, **kwargs) -> Tensor:
        """Align features to the temporal boundaries of the actions.

        As input, we get the output graphs of the temporal model, possibly at different depths.
        Then we align all the features according to the action boundaries specified in data.
        
        Usage example:
        
        ```python
        for data in dataloader:
            graphs = temporal_model(data)
            graphs.x = task.align(graphs, data)
        ```
                
        Parameters
        ----------
        graphs : Data
            output of the temporal model
        data : Data
            raw input data containing information for the alignemnt process

        Returns
        -------
        Tensor
            task-aligned features
        """
        features, pos, batch, segments, segments_batch, depth = graphs.x, graphs.pos, graphs.video, data.segments, data.segments_batch, graphs.depth
        
        # Align node positions with the ground truth segments and pool features.
        # We use start and end boundaries of the action segments (according to the ground truth)
        # to max (or mean) pool features for each action.
        
        # pos.shape = (N), batch.shape = (N), segments.shape = (M, 2), segments_batch.shape = (M)
        
        assert (depth == 0).all(), "only depth == 0 is supported atm"
        
        return align(features, pos, segments, segments_batch, batch)  # type: ignore
            
    def forward(self, graphs: Data, data: Data, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
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
        features, pos, batch, segments, segments_batch, depth = graphs.x, graphs.pos, graphs.video, data.segments, data.segments_batch, graphs.depth
        
        # Make sure to keep only features from high-resolution features for this task
        features, pos, batch = features[depth == 0], pos[depth == 0], batch[depth == 0]  # type: ignore
        
        # Step 1: Project features
        features = self.project(features)
        
        # Step 2: Align node positions with the ground truth segments.
        # We use start and end boundaries of the action segments (according to the ground truth)
        # to max (or mean) pool features for each action.
        # pos.shape = (N), batch.shape = (N), segments.shape = (M, 2), segments_batch.shape = (M)
        features = align(features, pos, segments, segments_batch, batch)

        # Step 3: Verb and noun classifiers
        verb_logits = self.verb_classifier(features)
        noun_logits = self.noun_classifier(features)
        
        return verb_logits, noun_logits, features

    def compute_loss(self, outputs: Tensor, graphs: Data, data: Data) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
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
        verb_loss = self.loss_fn(verb_logits, verb_labels)
        noun_loss = self.loss_fn(noun_logits, noun_labels)
        
        total_loss = self.loss_weight * (verb_loss + noun_loss)
        
        return total_loss, (verb_loss, noun_loss)


class EgoPackARTask(ARTask, EgoPackTask):
    """EgoPack Task for AR"""
    
    def __init__(self,
                 verb_classifier: Dict[str, Any],
                 noun_classifier: Dict[str, Any],
                 # EgoPack-specific configurations
                 fusion_level: Literal['features', 'logits'] = 'logits',
                 fusion_dropout: float = 0.1,
                 aux_tasks: Optional[List[str]] = None,
                 **kwargs
        ):
        """Initialize the EgoPack AR Task.

        Parameters
        ----------
        verb_classifier : Dict[str, Any]
            verb classifier configuration
        noun_classifier : Dict[str, Any]
            noun classifier configuration
        fusion_level : Literal['features', 'logits'], optional
            fusion level for EgoPack, by default 'logits'
        fusion_dropout : float, optional
            dropout in the EgoPack fusion step, by default 0.1
        aux_tasks : Optional[List[str]], optional
            list of auxiliary tasks, by default None
        """
        super().__init__(verb_classifier=verb_classifier, noun_classifier=noun_classifier, **kwargs)

        self.fusion_level: Literal['features', 'logits'] = fusion_level
        
        # Initialize classifiers for the auxiliary tasks
        if fusion_level == 'logits':
            assert aux_tasks is not None, 'Auxiliary tasks not provided'
            self.aux_verbs_classifier = nn.ModuleDict({task: hydra.utils.instantiate(verb_classifier, bias=False) for task in aux_tasks})
            self.aux_nouns_classifier = nn.ModuleDict({task: hydra.utils.instantiate(noun_classifier, bias=False) for task in aux_tasks})
        
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
        
        return align(features, pos, data.segments, data.segments_batch, batch)
            
    def forward(self, graphs: Data, data: Data, auxiliary_features: Dict[str, Tensor], **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
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
        features, pos, batch, segments, segments_batch, depth = graphs.x, graphs.pos, graphs.video, data.segments, data.segments_batch, graphs.depth
        
        # Make sure to keep only features from high-resolution features for this task
        features, pos, batch = features[depth == 0], pos[depth == 0], batch[depth == 0]  # type: ignore
        
        # Step 1: Project and align main task features
        features = self.project(features)
        features = align(features, pos, segments, segments_batch, batch)
        
        # Step 2 (mutually exclusive with step 4): Features fusion
        if self.fusion_level == 'features':
            # Features-level fusion: sum the main task features and the auxiliary features
            features = features + torch.stack(list(auxiliary_features.values())).mean(0)
        
        # Step 3: Main task classification
        verb_logits, noun_logits = (self.verb_classifier(features), self.noun_classifier(features))

        # Step 4 (mutually exclusive with step 2): Logits fusion
        if self.fusion_level == 'logits':
            # Forward the auxiliary features through their specific classifiers
            aux_verb_logits = [self.aux_verbs_classifier[task](f) for task, f in auxiliary_features.items()]
            aux_noun_logits = [self.aux_nouns_classifier[task](f) for task, f in auxiliary_features.items()]
            
            verb_logits = torch.stack([verb_logits, *aux_verb_logits]).mean(0)
            noun_logits = torch.stack([noun_logits, *aux_noun_logits]).mean(0)
        
        return (verb_logits, noun_logits, features)
    

@torch.jit.script  # type: ignore
def align(features: Tensor, pos: Tensor, segments: Tensor, segments_batch: Tensor, batch: Tensor) -> Tensor:
    """Pool features according to segments assignment.
    
    Unlike mean and max pool operations from torch geometric, this function supports
    overlapping segments.

    Parameters
    ----------
    features : Tensor
        input features (n x feat_size)
    pos : Tensor
        node position (n)
    segments : Tensor
        segments (segments x 2)
    segments_batch : Tensor
        batch tensor for the segments (segments)
    batch : Tensor
        batch tensor for the nodes (n)

    Returns
    -------
    Tensor
        aligned and pooled features
    """
    mask = segments_batch[:, None] == batch[None, :]  # node and gt are from the same video
    mask = torch.logical_and(mask, segments[:, :1] <= pos[None, :])
    mask = torch.logical_and(mask, segments[:, 1:] >= pos[None, :])
    
    count = mask.sum(1).clamp(min=1)  # number of nodes for segment
    
    # for each row i in mask, torch.argwhere(mask[i]) returns the indices that should 
    # be pooled for the corresponding ground truth
    features = mask[:, :, None] * features[None]

    return features.sum(1) / count[:, None]
