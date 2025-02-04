"""Ego4d Moment Queries Task"""

import logging
from typing import Tuple, Optional, Dict, List, Literal, Any

import hydra

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from torch_geometric import nn as gnn
from torch_geometric.data import Data
from torch_geometric.utils import unbatch

from models.tasks.task import Task, EgoPackTask


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MQTask(Task):
    """Moment Queries (MQ) task."""

    
    def __init__(self, 
                 name: str, 
                 input_size: int, 
                 features_size: int, 
                 num_classes: int,
                 classifier: Dict[str, Any],
                 regressor: Dict[str, Any],
                 depth: int = 8,
                 dropout: float = 0, 
                 cls_loss: Literal['focal', 'bce'] = 'focal',
                 pool_features: Optional[Literal['max', 'mean']] = None,
                 # Classification and regression loss weights
                 weight_cls: float = 1.0, weight_reg: float = 1.0,
                 loss_weight: float = 1.0,
                 **kwargs  # pylint: disable=unused-argument
        ):
        """Initialize the Ego4d MQ task.
        
        This task consumes a set of graphs with different temporal granularities that are
        produced as output of the temporal backbone.

        Parameters
        ----------
        name : str
            name of the task (e.g. 'ego4d/mq')
        input_size : int
            size of the input features
        features_size : int
            size of the task (hidden) features
        num_classes : int
            number of unique class labels
        classifier : Dict[str, Any]
            configuration of the classifier
        regressor : Dict[str, Any]
            configuration of the regressor
        depth : int, optional
            max temporal depth of the input features, by default 8
        dropout : float, optional
            dropout in the projection neck, by default 0
        cls_loss : Literal['focal', 'bce'], optional
            loss for the classification head, by default 'focal'
        pool_features : Optional[Literal['max', 'mean']], optional
            optionally concatenate mean/max pooled features from the entire video as classifier's input, by default None
        weight_cls : float, optional
            weight associated to the classification loss, by default 1.0
        weight_reg : float, optional
            weight associated to the regression loss, by default 1.0
        loss_weight : float, optional
            weight associated to the total loss, by default 1.0
        """
        logger.info("Initializing %s task with input size %d and features_size %d.", name, input_size, features_size)
        
        super().__init__(name, input_size, features_size, dropout)

        self.num_classes = num_classes
        
        # Loss configuration
        self.weight_cls = weight_cls
        self.weight_reg = weight_reg
        self.loss_weight = loss_weight
        self.cls_loss = cls_loss
        
        # Loss normalizer (taken from ActionFormer)
        self.loss_normalizer = 100
        self.loss_normalizer_momentum = 0.9
        
        # Initialize the classifier and regressor heads
        self.pool_features = pool_features
        head_input_size = (features_size * 2) if (pool_features in ['max', 'mean']) else features_size
        self.classifier: nn.Module = hydra.utils.instantiate(classifier, features_size=head_input_size)
        self.regressor: nn.Module = hydra.utils.instantiate(regressor)
        self.scales: nn.Parameter = nn.Parameter(torch.ones((depth,)), requires_grad=True)
            
    def forward(self, graphs: Data, data: Data, **kwargs) -> Tuple[Tensor, ...]:
        """Forward features through the projection neck and the task-specific heads.
        
        Beware, here graphs.batch identifies different temporal granularities, 
        NOT different videos. To access, different videos use graphs.video instead.

        Parameters
        ----------
        graphs : Data
            output of the temporal GNN
        data : Optional[Tensor], optional
            input data to the model (may be used to retrieve labels or other metadata on the input)

        Returns
        -------
        Tensor
            output logits (tensor shape: [batch_size, n_classes])
        """
        features: Tensor = graphs.x  # type: ignore
        
        # Important: use graphs.video rather than graphs.batch to identify different videos
        batch, depth = graphs.video, graphs.depth
        
        # Step 1: Project features
        features = self.project(features)
        # Split features into classification and regression features (initially the same)
        cls_features, reg_features = features, features

        # Step 2: Optionally pool classification features
        if self.pool_features in ['mean', 'max']:
            if self.pool_features == 'max':
                pooled_features = gnn.pool.global_max_pool(cls_features, batch)
            else:  # mean
                pooled_features = gnn.pool.global_mean_pool(cls_features, batch)

            pooled_features = torch.index_select(pooled_features, 0, batch)
            cls_features = torch.cat([cls_features, pooled_features], dim=1)

        # Step 3: Compute classification and regression predictions
        cls_logits = self.forward_cls(cls_features)
        reg_preds = self.forward_reg(reg_features, depth)
        return cls_logits, reg_preds

    def forward_cls(self, features: Tensor) -> Tensor:
        """Forward the features through the classifiers and the auxiliary classifiers, if any.

        Parameters
        ----------
        features : Tensor
            input features (tensor shape: [batch_size, features_size])

        Returns
        -------
        Tensor
            output logits (tensor shape: [batch_size, 2])
        """
        
        return self.classifier(features)
    
    def forward_reg(self, features: Tensor, depth: Tensor) -> Tensor:
        """Forward the features through the regression head and the auxiliary heads, if any.

        Parameters
        ----------
        features : Tensor
            input features (tensor shape: [batch_size, features_size])s
        depth : Optional[Tensor], optional
            depth in the GNN hierarchy, by default None (no hierarchy)

        Returns
        -------
        Tensor
            output logits (tensor shape: [batch_size, 2])
        """

        preds = self.regressor(features)
        return preds * torch.index_select(self.scales, 0, depth)[:, None]

    def compute_loss(self, outputs: Tuple[Tensor, Tensor], graphs: Data, data: Data) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Compute the loss function for the MQ task.
        
        This code is taken/inspired from ActionFormer.

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
        cls_logits, reg_preds = outputs
        labels, segments, labels_batch = data.labels, data.segments, data.labels_batch

        # Some optional sanity checks
        # assert cls_logits.shape[0] == graphs.pos.shape[0], "Logits and positions must have the same length."
        # assert cls_logits.shape[0] == graphs.batch.shape[0], "Logits and batch must have the same length."
        # assert cls_logits.shape[0] == graphs.extent.shape[0], "Logits and extent must have the same length."
        # assert len(graphs.labels) == len(graphs.segments) == torch.unique(graphs.video_batch).size(0), "Labels and segments must have the same length."
        
        cls_losses: List[Tensor] = []
        reg_losses: List[Tensor] = []
        
        n: int = 0
        
        # Count the number of unique videos in the input
        n_videos: int = torch.unique(graphs.video).size(0)
        
        # Separate labels and segments by video id. Note that here we can use the
        # unbatch utility from torch_geometric only because labels_batch is sorted but
        # it may NOT be the case in general
        labels: List[Tensor] = unbatch(labels, labels_batch)  # type: ignore
        segments: List[Tensor] = unbatch(segments, labels_batch)  # type: ignore

        # Compute the classification loss separately for each video
        for video in range(n_videos):
            mask: torch.BoolTensor = graphs.video == video
            
            # For each video take the logits, the regression predictions and the pos, extent, depth attribute
            graph_cls_logits: Tensor = cls_logits[mask]
            graph_reg_preds: Tensor = reg_preds[mask]
            graph_pos: Tensor = graphs.pos[mask]  # type: ignore
            graph_extent: Tensor = graphs.extent[mask]  # type: ignore
            graph_depth: Tensor = graphs.depth[mask]  # type: ignore
            
            # Take the labels and segments of the video-th video
            graph_labels: Tensor = labels[video]
            graph_segments: Tensor = segments[video]
            
            assert graph_cls_logits.shape[0] == graph_reg_preds.shape[0], "Logits and predictions must have the same length."
            assert graph_cls_logits.shape[0] == graph_pos.shape[0], "Logits and positions must have the same length."
            assert graph_cls_logits.shape[0] == graph_extent.shape[0], "Logits and extent must have the same length."
            
            # Number of predictions in the video (equal to the sum of cardinality of all the graphs for the current video)
            num_pts: int = graph_cls_logits.shape[0]
            # Number of ground truth labels
            num_gts: int = graph_labels.shape[0]
        
            with torch.no_grad():
                # assert torch.all(graph_segments >= 0), "Segment boundaries must be positive."
                # assert torch.all(graph_segments[:, 1] > graph_segments[:, 0]), "Segments must have positive duration."
                
                # Replicate the ground truth segments for all the points in this video
                gt_segs = graph_segments[None].expand(num_pts, num_gts, 2)
                
                # reg_targets represents the distance of the points from the left 
                # and right extrema of the ground truth segments
                left = graph_pos[:, None] - gt_segs[:, :, 0]
                right = gt_segs[:, :, 1] - graph_pos[:, None]
                graph_reg_targets = torch.stack((left, right), dim=-1)
                
                # This gt is valid iff: i) the point is within the segment, ii) the segment is within the extent
                inside_gt_seg_mask = graph_reg_targets.min(-1)[0] > 0  # (num_pts, num_segments)
                min_regr_range, max_regr_range = 2 ** graph_depth, 2 ** (graph_depth + 2)
                min_regr_range = torch.where(graph_depth == 0, 0, min_regr_range)
                max_regr_range = torch.where(graph_depth == graph_depth.max(), 10000, max_regr_range)

                # For each prediction, inside_regress_range identifies all the target that
                # are within the valid regression range
                inside_regress_range = torch.logical_and(
                    graph_reg_targets.max(-1)[0] >= min_regr_range.unsqueeze(1),
                    graph_reg_targets.max(-1)[0] <= max_regr_range.unsqueeze(1)
                )  # (num_pts, num_segments)
                
                lens = graph_segments[:, 1] - graph_segments[:, 0]
                lens = lens[None, :].repeat(num_pts, 1)
                lens.masked_fill_(torch.logical_not(inside_gt_seg_mask), float('inf'))
                lens.masked_fill_(torch.logical_not(inside_regress_range), float('inf'))
                min_len, min_len_inds = lens.min(dim=1)
                
                min_len_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))).to(graph_reg_targets.dtype)

                # pylint: disable=not-callable
                labels_one_hot = F.one_hot(graph_labels, self.num_classes).to(graph_reg_targets.dtype)
                
                # As a classification target we take the action with shortest duration
                # among the ones that are in the correct regression range
                graph_cls_targets = min_len_mask.float() @ labels_one_hot
                graph_cls_targets.clamp_(min=0.0, max=1.0)

                graph_reg_targets = graph_reg_targets[range(num_pts), min_len_inds] / (2 ** graph_depth).unsqueeze(1)
                graph_pos_mask = graph_cls_targets.sum(-1) > 0

            if self.cls_loss == 'bce':
                cls_losses.append(F.binary_cross_entropy_with_logits(graph_cls_logits, graph_cls_targets.detach(), reduction='sum').unsqueeze(0))
            else:
                cls_losses.append(sigmoid_focal_loss(graph_cls_logits, graph_cls_targets.detach(), reduction='sum').unsqueeze(0))

            if graph_pos_mask.sum() == 0:
                # Skip this value
                continue
            
            reg_loss = ctr_diou_loss_1d(graph_reg_preds[graph_pos_mask].clip(min=0), graph_reg_targets[graph_pos_mask].detach(), reduction='none').unsqueeze(0)
            reg_losses.append(reg_loss)
            n += int(graph_pos_mask.sum())

        if self.training:
            # do not update the loss normalizer during evaluation
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (1 - self.loss_normalizer_momentum) * max(n, 1)
            cls_loss: Tensor = torch.cat(cls_losses).sum() / self.loss_normalizer
            reg_loss: Tensor = torch.cat(reg_losses, 1).sum() / self.loss_normalizer
        else:
            if n == 0:
                cls_loss, reg_loss = Tensor([0]).squeeze(), Tensor([0]).squeeze()
            else:
                cls_loss: Tensor = torch.cat(cls_losses).sum() / max(n, 1)
                reg_loss: Tensor = torch.cat(reg_losses).sum() / max(n, 1)
        
        loss = self.weight_cls * cls_loss + self.weight_reg * reg_loss
        return self.loss_weight * loss, (cls_loss, reg_loss)


class EgoPackMQTask(MQTask, EgoPackTask):
    """EgoPack Task for MQ"""
    
    def __init__(self, fusion_dropout: float = 0.1, project_before_fusion: bool = True, **kwargs):
        """Initialize the EgoPack MQ Task.

        Parameters
        ----------
        fusion_dropout : float, optional
            dropout in the EgoPack fusion step, by default 0.1
        project_before_fusion : bool, optional
            project auxiliary features before EgoPack fusion, by default False
        """
        super().__init__(**kwargs)

        self.project_before_fusion: bool = project_before_fusion
        self.fusion_dropout: float = fusion_dropout
            
    # pylint: disable=unused-argument,arguments-differ
    def forward(self, graphs: Data, data: Data, auxiliary_features: Dict[str, Tensor], **kwargs) -> Tuple[Tensor, ...]:
        """Forward features through the projection module and the classifiers.

        Parameters
        ----------
        graphs : Data
            output of the temporal GNN
        data : Tensor
            input data to the model (may be used to retrieve labels or other metadata on the input)
        auxiliary_features : Dict[str, Tensor]
            auxiliary features from other tasks

        Returns
        -------
        Tensor
            output logits (tensor shape: [batch_size, n_classes])
        """
        features: Tensor = graphs.x  # type: ignore
        
        # Important: use graphs.video rather than graphs.batch to identify different videos
        batch, depth = graphs.video, graphs.depth
        
        # Step 1: Project features
        features = self.project(features)
        
        # Step 2: Add the EgoPack features
        features = torch.stack([features] + [self.project(f) if self.project_before_fusion else f for f in auxiliary_features.values()])
        features = F.dropout(features, self.fusion_dropout, training=self.training).mean(0)
        
        # Split features into classification and regression features (initially the same)
        cls_features, reg_features = features, features

        # Step 3: Optionally pool classification features
        if self.pool_features in ['mean', 'max']:
            if self.pool_features == 'max':
                pooled_features = gnn.pool.global_max_pool(cls_features, batch)
            else:  # mean
                pooled_features = gnn.pool.global_mean_pool(cls_features, batch)

            pooled_features = torch.index_select(pooled_features, 0, batch)
            cls_features = torch.cat([cls_features, pooled_features], dim=1)

        # Step 4: Compute classification and regression predictions
        cls_logits = self.forward_cls(cls_features)
        reg_preds = self.forward_reg(reg_features, depth)
        return cls_logits, reg_preds
        
        
def unbatch_all(*data: Tensor, batch: Tensor) -> Tuple[Tensor]:
    """Unbatch all tensors in the list."""
    return tuple(zip(*[unbatch(tensor, batch) for tensor in data]))


def ctr_diou_loss_1d(
    input_offsets: Tensor,
    target_offsets: Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float().clamp(min=0)
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"
    
    input_offsets[:, 0] = input_offsets[:, 0] * (-1.0)
    target_offsets[:, 0] = target_offsets[:, 0] * (-1.0)

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.max(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = torch.clamp(rkis - lkis, min=0)
    unionk = (rp - lp) + (rg - lg) - intsctk
    iou = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.min(lp, lg)
    rc = torch.max(rp, rg)

    # offset between centers
    # centers = (pred_boxes[:, 0] + pred_boxes[:, 1])[:, None] / 2
    # gt_centers = (gt_boxes[:, 0] + gt_boxes[:, 1]) / 2
    # center_dist = torch.pow(centers - gt_centers, 2)
    centers = (lp + rp) / 2
    gt_centers = (lg + rg) / 2
    
    # print(iou.mean())

    # Calculate dIoU
    loss = 1.0 - iou + (centers - gt_centers).pow(2) / (rc - lc).pow(2).clamp(min=eps)

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script  # type: ignore
def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def iou_1d(intervals1, intervals2):
    """
    Computes Intersection over Union for multiple pairs of 1D intervals.

    Args:
        intervals1 (Tensor): A tensor of shape (N, 2) representing the first set of intervals.
        intervals2 (Tensor): A tensor of shape (N, 2) representing the second set of intervals.

    Returns:
        Tensor: A tensor of shape (N,) containing the IoU for each pair of intervals.
    """
    intervals1[:, 0] = intervals1[:, 0] * (-1.0)
    intervals2[:, 0] = intervals2[:, 0] * (-1.0)

    start_intersections = torch.max(intervals1[:, 0], intervals2[:, 0])
    end_intersections = torch.min(intervals1[:, 1], intervals2[:, 1])
    intersection_lengths = torch.clamp(end_intersections - start_intersections, min=0)

    union_lengths = (intervals1[:, 1] - intervals1[:, 0]) + (intervals2[:, 1] - intervals2[:, 0]) - intersection_lengths
    ious = intersection_lengths / (union_lengths + 1e-7)
    return ious
