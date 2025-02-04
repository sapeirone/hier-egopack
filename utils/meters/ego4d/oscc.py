import torch

from torch_geometric.data.data import Data
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import MulticlassAccuracy

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
                 prefix: str = 'ego4d/oscc', step_metric: str = "train/step",
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
                 prefix: str = 'ego4d/oscc', step_metric: str = "val/step",
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__(mode, prefix, step_metric, device)

        # Accuracies for verbs
        self.accuracy = Metric("acc", MulticlassAccuracy(num_classes=2, average="micro", ignore_index=-1).to(device), "max", "maximize")

    @torch.no_grad()
    def update(self, outputs, losses: Tuple[torch.Tensor, ...], graphs: Data, data: Data) -> None:
        super().update(losses)
        logits, _ = outputs
        
        self.accuracy.metric.update(logits, data.labels)
        
    @torch.no_grad()
    def logs(self, *args) -> None: 
        accuracy = 100 * self.accuracy.metric.compute()
        loss = self.loss.metric.compute()
        print_logs([f"Accuracy: {accuracy:.2f}. Loss={loss:.4f}."])
