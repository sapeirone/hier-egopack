import torch

from torch_geometric.data.data import Data
from torch_geometric.utils import unbatch

from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAUROC

from typing import List, Tuple, Literal

from utils.meters.ego4d.base import BaseMeter, Metric

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def print_logs(logs: List[str]) -> None:
    for log in logs:
        logger.info(log)
        

class TrainMeter(BaseMeter):
    
    def __init__(self,
                 mode: Literal['train', 'val'] = 'train',
                 prefix: str = 'ego4d/pnr', step_metric: str = "train/step",
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__(mode, prefix, step_metric, device)
        
        # Define loss metrics
        self.loss = Metric("loss", MeanMetric().to(device), "min", "minimize")
    
    @torch.no_grad()
    def update(self, loss: Tuple[torch.Tensor, ...]) -> None:
        # Update the loss meters
        oscc_loss, *_ = loss
        self.loss.metric.update(oscc_loss)
       
    @torch.no_grad() 
    def logs(self, *args):
        loss = self.loss.metric.compute()
        print_logs([f"Loss={loss:.4f}."])


class EvalMeter(TrainMeter):
    
    def __init__(self, 
                 mode: Literal['train', 'val'] = 'val',
                 prefix: str = 'ego4d/pnr', step_metric: str = "val/step",
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__(mode, prefix, step_metric, device)
        
        self.auroc = Metric("auroc", BinaryAUROC().to(device), "max", "maximize")
        self.loc_errors = Metric("loc_errors", MeanMetric().to(device), "min", "minimize")

    @torch.no_grad()
    def update(self, outputs, losses: Tuple[torch.Tensor, ...], graphs: Data, data: Data) -> None:
        super().update(losses)
        
        logits, *_ = outputs
        
        self.auroc.metric.update(logits, data.labels)
        
         # update verbs metrics
        for preds, sf, pf, cf in zip(unbatch(logits.squeeze(), data.batch), data.start_frame, data.pnr_frame, data.candidate_frames.unbind(0)):
            
            if torch.all(preds < 0.7):
                keyframe_loc_pred = torch.argmax(preds).item()
            else:
                keys = torch.argwhere(preds > 0.7) - 7
                keyframe_loc_pred = 7 + keys[keys.abs().argmin()]    

            keyframe_loc_pred_mapped = cf[keyframe_loc_pred] - sf
            keyframe_loc_pred_mapped = keyframe_loc_pred_mapped.item()

            # absolute distance of the pnr frame from the start of the clip
            gt = pf.item() - sf.item()

            err_frame = abs(keyframe_loc_pred_mapped - gt)
            err_sec = err_frame/30
            self.loc_errors.metric.update(err_sec)
        
    @torch.no_grad()
    def logs(self, *args) -> None: 
        auroc = self.auroc.metric.compute()
        loc_error = self.loc_errors.metric.compute()

        print_logs([f"AUROC={auroc:.2f}. loc_error={loc_error:.4f}."])
