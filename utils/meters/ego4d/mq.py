import os
from os import path as osp

import wandb

import itertools as it

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import torch

import torch_geometric.nn as gnn
from torch_geometric.data.data import Data
from torchmetrics.aggregation import MeanMetric

from torchmetrics.classification import MultilabelAveragePrecision

from typing import List, Tuple, Literal, Dict, Optional

from utils.meters.ego4d.base import BaseMeter, Metric

from libs.utils.nms import batched_nms

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def print_logs(logs: List[str]) -> None:
    for log in logs:
        logger.info(log)
        

class TrainMeter(BaseMeter):
    
    def __init__(self, 
                 mode: Literal['train', 'val'] = 'train',
                 prefix: str = 'ego4d/mq', step_metric: str = "train/step",
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__(mode, prefix, step_metric, device)
        
        # Define loss metrics
        self.cls_loss = Metric("cls_loss", MeanMetric().to(device), "min", "minimize")
        self.reg_loss = Metric("reg_loss", MeanMetric().to(device), "min", "minimize")
    
    @torch.no_grad()
    def update(self, loss: Optional[Tuple[torch.Tensor, ...]]) -> None:
        # Update the loss meters
        if loss is not None:
            cls_loss, reg_loss, *_ = loss
            self.cls_loss.metric.update(cls_loss)
            self.reg_loss.metric.update(reg_loss)
       
    @torch.no_grad() 
    def logs(self):
        cls_loss, reg_loss = self.cls_loss.metric.compute(), self.reg_loss.metric.compute()
        print_logs([f"CLS Loss = {cls_loss:.4f}, REG Loss = {reg_loss:.4f}"])


class EvalMeter(TrainMeter):
    
    def __init__(self, 
                 num_labels: int = 110, iou_thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
                 mode: Literal['train', 'val'] = 'val',
                 prefix: str = 'ego4d/oscc', step_metric: str = "val/step",
                 fast: bool = False,
                 nms_sigma: float = 0.9,
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__(mode, prefix, step_metric, device)
        
        self.num_labels = num_labels
        
        self.fast = fast

        # Values from ActionFormer
        self.pre_nms_thresh = 0.001
        self.pre_nms_topk = 5000
        self.duration_thresh = 0.05
        
        # NMS config (from ActionFormer)
        self.nms_method = 'soft'
        self.multiclass_nms = True
        self.nms_sigma = nms_sigma
        logger.info("Using nms_sigma={}".format(self.nms_sigma))
        
        self.voting_thresh = 0.95
        self.iou_threshold = 0.1
        self.min_score = 0.001
        self.max_seg_num = 2000
        self.iou_thresholds = iou_thresholds
        
        self.map = Metric("map", MultilabelAveragePrecision(num_labels=num_labels, average="macro", ignore_index=-1).to(device), "max", "maximize")
        self.weighted_map = Metric("weighted_map", MultilabelAveragePrecision(num_labels=num_labels, average="weighted", ignore_index=-1).to(device), "max", "maximize")
        
        self.class_ap = MultilabelAveragePrecision(num_labels=num_labels, average="none", ignore_index=-1).to(device)
        
        self.det_map = Metric("detection/mAP", MultilabelAveragePrecision(num_labels=num_labels, average="macro", ignore_index=-1).to(device), "max", "maximize")
        self.det_weighted_map = Metric("detection/weighted_mAP", MultilabelAveragePrecision(num_labels=num_labels, average="weighted", ignore_index=-1).to(device), "max", "maximize")
        
        self.results = {
            'video-id': [],
            't-start' : [],
            't-end': [],
            'label': [],
            'score': []
        }
    
        self.gts = {
            'video-id': [],
            't-start' : [],
            't-end': [],
            'label': [],
            'coverage': [],
            'length': [],
            'num-instances': []
        }
        
        self.precomputed_metrics = None
        
    def define_metrics(self):
        super().define_metrics()

        for iou in self.iou_thresholds:
            wandb.define_metric(f"val/ego4d/mq/mAP@IoU={iou:.2f}", step_metric="val/step", summary="max", goal="maximize")
            wandb.define_metric(f"val/ego4d/mq/mRecall1x@IoU={iou:.2f}", step_metric="val/step", summary="max", goal="maximize")
            wandb.define_metric(f"val/ego4d/mq/mRecall5x@IoU={iou:.2f}", step_metric="val/step", summary="max", goal="maximize")

    def reset(self):
        super().reset()
        # Cleanup results and ground truths
        self.results = {key: [] for key in self.results.keys()}
        self.gts = {key: [] for key in self.gts.keys()}
        self.precomputed_metrics = None

    @torch.no_grad()
    def update(self, 
               outputs: Tuple[torch.Tensor, torch.Tensor], 
               losses: Tuple[torch.Tensor, torch.Tensor], 
               graphs: Data, data: Data) -> None:
        super().update(losses)
        
        assert torch.unique(data.batch).size(0) == 1, "Batch size must be 1 for validation."
        
        cls_logits, reg_preds = outputs
        
        video_results = self.inference_single_video(cls_logits, reg_preds, 
                                                    graphs.pos, graphs.extent, 
                                                    data.clip_uid, data.fps, data.duration, data.feat_stride, data.feat_num_frames)
        
        self.results['video-id'] += data.clip_uid * video_results['segments'].shape[0]
        self.results['t-start'].append(video_results['segments'][:, 0])
        self.results['t-end'].append(video_results['segments'][:, 1])
        self.results['label'].append(video_results['labels'])
        self.results['score'].append(video_results['scores'])
        
        if hasattr(data, 'labels') and data.labels is not None:
            # compute video-level labels
            labels = torch.nn.functional.one_hot(data.labels.unique(), num_classes=self.num_labels).sum(dim=0)  # type: ignore
            labels = torch.clip(labels, max=1).unsqueeze(0).long()
            
            video_logits = gnn.pool.global_max_pool(cls_logits, graphs.video)
            
            self.det_map.metric.update(video_logits, labels)
            self.det_weighted_map.metric.update(video_logits, labels)
            
            self.class_ap.update(video_logits, labels)

            for clip_uid, (start, end), label, coverage, length in zip(it.cycle(data.clip_uid), data.segments.cpu().tolist(), data.labels.cpu().tolist(), data.coverage.cpu().tolist(), data.length.cpu().tolist()):
                self.gts['video-id'].append(clip_uid)
                self.gts['t-start'].append(start)
                self.gts['t-end'].append(end)
                self.gts['label'].append(label)
                self.gts['coverage'].append(coverage)
                self.gts['length'].append(length)
                self.gts['num-instances'].append(data.num_instances.cpu().item())
            
    def __precompute_metrics(self):
        if self.precomputed_metrics is None:
            gts = pd.DataFrame(self.gts)
            
            self.results['t-start'] = torch.cat(self.results['t-start'], 0)  # type: ignore
            self.results['t-end'] = torch.cat(self.results['t-end'], 0)  # type: ignore
            self.results['label'] = torch.cat(self.results['label'], 0)  # type: ignore
            self.results['score'] = torch.cat(self.results['score'], 0)  # type: ignore
            
            self.precomputed_metrics = ANETdetection(gts).evaluate(self.results, verbose=False)
            
    def generate_inference_outputs(self):
        """Generate inference outputs"""
        
        results = pd.DataFrame({
            'video-id': self.results['video-id'],
            'xmin': torch.cat(self.results['t-start'], 0).tolist(),
            'xmax': torch.cat(self.results['t-end'], 0).tolist(),
            'label': torch.cat(self.results['label'], 0).tolist(),
            'score': torch.cat(self.results['score'], 0).tolist(),
        })
        
        os.makedirs('tmp', exist_ok=True)
        for clip_id, predictions in results.groupby('video-id'):
            predictions.to_csv(f"tmp/{clip_id}.csv")

    @torch.no_grad()
    def logs(self, artifacts_path: Optional[str] = None, *args) -> None:
        
        if not self.fast: 
            self.__precompute_metrics()

            average_mAP, *_ = self.precomputed_metrics  # type: ignore
            logger.info("ActionFormer mAP: {:.4f}.".format(average_mAP))

        logger.info(f"Detection: mAP={100*self.det_map.metric.compute():.2f}, weighted_mAP={100*self.det_weighted_map.metric.compute():.2f}.")
        logger.info(f"cls_loss={self.cls_loss.metric.compute():.4f}, reg_loss={self.reg_loss.metric.compute():.4f}.")
        
    def wandb_logs(self, step: int, *args, **kwargs):
        if not self.fast:
            self.__precompute_metrics()

            average_mAP, iou_thresholds, mAP, mRecall = self.precomputed_metrics  # type: ignore
            
            class_ap = self.class_ap.compute()
            
            data = [[str(label), val] for (label, val) in enumerate(class_ap)]  # type: ignore
            classwise_ap_table = wandb.Table(data=data, columns = ["label", "AP"])
            
            wandb.log({
                f"{self.step_metric}": step,
                f"{self.mode}/{self.prefix}/mAP": average_mAP,
                f"{self.mode}/{self.prefix}/cls_loss": self.cls_loss.metric.compute(),
                f"{self.mode}/{self.prefix}/reg_loss": self.reg_loss.metric.compute(),
                f"{self.mode}/{self.prefix}/detection/mAP": self.det_map.metric.compute(),
                f"{self.mode}/{self.prefix}/detection/weighted_mAP": self.det_weighted_map.metric.compute(),
                f"{self.mode}/{self.prefix}/detection/AP" : wandb.plot.bar(classwise_ap_table, "label", "AP", title="Class-wise Average Precision"),   # type: ignore
                **{f"{self.mode}/{self.prefix}/mAP@IoU={iou:.2f}": mAP[i] for i, iou in enumerate(iou_thresholds)},
                **{f"{self.mode}/{self.prefix}/mRecall1x@IoU={iou:.2f}": mRecall[i, 0] for i, iou in enumerate(iou_thresholds)},
                **{f"{self.mode}/{self.prefix}/mRecall5x@IoU={iou:.2f}": mRecall[i, 1] for i, iou in enumerate(iou_thresholds)}
            })
        else:
            wandb.log({
                f"{self.step_metric}": step,
                f"{self.mode}/{self.prefix}/cls_loss": self.cls_loss.metric.compute(),
                f"{self.mode}/{self.prefix}/reg_loss": self.reg_loss.metric.compute(),
                f"{self.mode}/{self.prefix}/detection/mAP": self.det_map.metric.compute(),
                f"{self.mode}/{self.prefix}/detection/weighted_mAP": self.det_weighted_map.metric.compute(),
            })

    @torch.no_grad()
    def _inference(
        self,
        all_cls_logits: torch.Tensor, 
        all_reg_preds: torch.Tensor, 
        all_pos: torch.Tensor, 
        all_extent: torch.Tensor
    ):
        
        all_pred_segs, all_pred_probs, all_cls_idxs = [], [], []
        
        for depth in all_extent.unique():
            cls_logits = all_cls_logits[all_extent == depth]
            reg_preds = all_reg_preds[all_extent == depth]
            pos = all_pos[all_extent == depth]
            extent = all_extent[all_extent == depth]
        
            # sigmoid normalization for output logits
            pred_prob = cls_logits.sigmoid().flatten() # (N, C) -> (N * C)

            # 1. Filter predictions above a threshold
            keep_idxs1 = (pred_prob > self.pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring predictions
            num_topk = min(self.pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs =  topk_idxs // self.num_labels
            cls_idxs = topk_idxs % self.num_labels

            # 3. gather predicted offsets
            offsets = reg_preds[pt_idxs]
            extent = extent[pt_idxs]
            pts = pos[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts - offsets[:, 0] * extent
            seg_right = pts + offsets[:, 1] * extent
            lens = offsets[:, 1] - offsets[:, 0]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            keep_idxs2 = (seg_right - seg_left) > (self.duration_thresh)
            
            all_pred_segs.append(pred_segs[keep_idxs2])
            all_pred_probs.append(pred_prob[keep_idxs2])
            all_cls_idxs.append(cls_idxs[keep_idxs2])

        return torch.cat(all_pred_segs, 0), torch.cat(all_pred_probs, 0), torch.cat(all_cls_idxs, 0)
    
    @torch.no_grad()
    def inference_single_video(self, cls_logits, reg_preds, pos, extent, video_uid, fps, duration, feat_stride, feat_num_frames) -> Dict[str, torch.Tensor]:
        # inference on a single video (should always be the case)
        pred_segs, pred_prob, cls_idxs = self._inference(cls_logits, reg_preds, pos, extent)
        results = {
            'segments': pred_segs,
            'scores': pred_prob,
            'labels': cls_idxs,
            # pass through video meta info
            'video_id': video_uid,
            'fps': fps,
            'duration': duration,
            'feat_stride': feat_stride,
            'feat_num_frames': feat_num_frames
        }

        # step 3: postprocssing
        results = self._postprocessing([results])[0]

        return results
    
    @torch.no_grad()
    def _postprocessing(self, results: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Postprocess the model predictions using NMS.
        
        Code taken from the ActionFormer repository.

        Parameters
        ----------
        results : dict
            results from the inference method

        Returns
        -------
        List[Dict[str, torch.Tensor]]
            _description_
        """
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            vlen = results_per_vid['duration'].cpu()
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            if self.nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.iou_threshold,
                    self.min_score,
                    self.max_seg_num,
                    use_soft_nms = (self.nms_method == 'soft'),
                    multiclass = self.multiclass_nms,
                    sigma = self.nms_sigma,
                    voting_thresh = self.voting_thresh
                )
            # # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                # segs = (segs * stride) / fps
                # truncate all boundaries within [0, duration]
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
            
            # 4: repack the results
            processed_results.append({
                'video_id' : vidx,
                'segments' : segs,
                'scores'   : scores,
                'labels'   : labels
            })

        return processed_results




####################################################
# All the code below was adapted from ActionFormer #
####################################################

class ANETdetection(object):

    def __init__(
        self,
        gts,
        split=None,
        tiou_thresholds=np.linspace(0.1, 0.5, 5),
        top_k=(1, 5),
        num_workers=8,
    ):

        self.tiou_thresholds = tiou_thresholds
        self.top_k = top_k
        self.ap = None
        self.num_workers = num_workers

        # Import ground truth and predictions
        self.split = split
        self.ground_truth = gts

        # remove labels that does not exists in gt
        self.activity_index = {j: i for i, j in enumerate(sorted(self.ground_truth['label'].unique()))}
        self.ground_truth['label']=self.ground_truth['label'].replace(self.activity_index)

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            res = prediction_by_label.get_group(cidx).reset_index(drop=True)
            return res
        except:
            # print('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self, preds):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = preds.groupby('label')

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx] = results[i]  # type: ignore

        return ap

    def wrapper_compute_topkx_recall(self, preds):
        """Computes Top-kx recall for each class in the subset.
        """
        recall = np.zeros((len(self.tiou_thresholds), len(self.top_k), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = preds.groupby('label')

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_topkx_recall_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
                top_k=self.top_k,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            recall[...,cidx] = results[i]  # type: ignore

        return recall

    def evaluate(self, preds, verbose=True):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """

        # move to pd dataframe
        # did not check dtype here, can accept both numpy / pytorch tensors
        preds = pd.DataFrame({
            'video-id' : preds['video-id'],
            't-start' : preds['t-start'].tolist(),
            't-end': preds['t-end'].tolist(),
            'label': preds['label'].tolist(),
            'score': preds['score'].tolist()
        })
        # always reset ap
        self.ap = None

        # make the label ids consistent
        preds['label'] = preds['label'].replace(self.activity_index)

        # compute mAP
        self.ap = self.wrapper_compute_average_precision(preds)
        self.recall = self.wrapper_compute_topkx_recall(preds)
        mAP = self.ap.mean(axis=1)
        mRecall = self.recall.mean(axis=2)
        average_mAP = mAP.mean()

        # print results
        if verbose:
            # print the results
            print('[RESULTS] Action detection results')
            block = ''
            for tiou, tiou_mAP, tiou_mRecall in zip(self.tiou_thresholds, mAP, mRecall):
                block += '\n|tIoU = {:.2f}: '.format(tiou)
                block += 'mAP = {:>4.2f} (%) '.format(tiou_mAP*100)
                for idx, k in enumerate(self.top_k):
                    block += 'Recall@{:d}x = {:>4.2f} (%) '.format(k, tiou_mRecall[idx]*100)
            print(block)
            print('Average mAP: {:>4.2f} (%)'.format(average_mAP*100))

        # return the results
        return average_mAP, self.tiou_thresholds, mAP, mRecall


def compute_average_precision_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5)
):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly ground truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap


def compute_topkx_recall_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5),
    top_k=(1, 5),
):
    """Compute recall (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    top_k: tuple, optional
        Top-kx results of a action category where x stands for the number of 
        instances for the action category in the video.
    Outputs
    -------
    recall : float
        Recall score.
    """
    if prediction.empty:
        return np.zeros((len(tiou_thresholds), len(top_k)))

    # Initialize true positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(top_k)))
    n_gts = 0

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')
    prediction_gbvn = prediction.groupby('video-id')

    for videoid, _ in ground_truth_gbvn.groups.items():
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        n_gts += len(ground_truth_videoid)
        try:
            prediction_videoid = prediction_gbvn.get_group(videoid)
        except Exception as e:
            continue

        this_gt = ground_truth_videoid.reset_index()
        this_pred = prediction_videoid.reset_index()

        # Sort predictions by decreasing score order.
        score_sort_idx = this_pred['score'].values.argsort()[::-1]
        top_kx_idx = score_sort_idx[:max(top_k) * len(this_gt)]
        tiou_arr = k_segment_iou(this_pred[['t-start', 't-end']].values[top_kx_idx],
                                 this_gt[['t-start', 't-end']].values)
            
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for kidx, k in enumerate(top_k):
                tiou = tiou_arr[:k * len(this_gt)]
                tp[tidx, kidx] += ((tiou >= tiou_thr).sum(axis=0) > 0).sum()

    recall = tp / n_gts

    return recall


def k_segment_iou(target_segments, candidate_segments):
    return np.stack(
        [segment_iou(target_segment, candidate_segments) \
            for target_segment in target_segments]
    )


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap