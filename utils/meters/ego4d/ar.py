import wandb

import torch

from torch_geometric.data.data import Data
from torchmetrics.aggregation import MeanMetric, CatMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAveragePrecision,
)
from torchmetrics.functional.classification import multiclass_accuracy

from typing import Tuple, List, Literal, Optional, Dict, Any

from utils.meters.ego4d.base import BaseMeter, Metric

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def print_logs(logs: List[str]) -> None:
    for log in logs:
        logger.info(log)
        

class TrainMeter(BaseMeter):
    
    def __init__(self, 
                 num_verbs: int, num_nouns: int,
                 mode: Literal['train', 'val'] = 'train',
                 prefix: str = 'ego4d/ar', step_metric: str = "train/step",
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__(mode, prefix, step_metric, device)
        
        self.idx_verbs, self.idx_nouns = 0, 1

        # Get the index of verb and nouns ground truth labels
        self.verb_labels = num_verbs
        self.noun_labels = num_nouns
        
        # Define loss metrics
        self.verbs_loss = Metric("verbs/loss", MeanMetric().to(device), "min", "minimize")
        self.nouns_loss = Metric("nouns/loss", MeanMetric().to(device), "min", "minimize")
    
    @torch.no_grad()
    def update(self, loss: Tuple[torch.Tensor, ...]) -> None:
        # Update the loss meters
        verb_loss, noun_loss = loss
        verb_loss, noun_loss = verb_loss.detach().to(self.device), noun_loss.detach().to(self.device)
        
        self.verbs_loss.metric.update(verb_loss)
        self.nouns_loss.metric.update(noun_loss)
       
    @torch.no_grad() 
    def logs(self, *args):
        verbs_loss, nouns_loss = self.verbs_loss.metric.compute(), self.nouns_loss.metric.compute()
        print_logs([f"Verbs: loss={verbs_loss:.4f}.", f"Nouns: loss={nouns_loss:.4f}."])


class EvalMeter(TrainMeter):
    
    def __init__(self, num_verbs: int, num_nouns: int,
                 mode: Literal['train', 'val'] = 'val',
                 prefix: str = 'ego4d/ar', step_metric: str = "val/step",
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__(num_verbs, num_nouns, mode, prefix, step_metric, device)
        
        self.num_verbs, self.num_nouns = num_verbs, num_nouns

        # Accuracies for verbs
        self.verbs_top1 = Metric("verbs/top1", MulticlassAccuracy(num_verbs, top_k=1, average="micro", ignore_index=-1).to(self.device), "max", "maximize")
        self.verbs_top2 = Metric("verbs/top2", MulticlassAccuracy(num_verbs, top_k=2, average="micro", ignore_index=-1).to(self.device), "max", "maximize")
        self.verbs_top3 = Metric("verbs/top3", MulticlassAccuracy(num_verbs, top_k=3, average="micro", ignore_index=-1).to(self.device), "max", "maximize")
        self.verbs_top5 = Metric("verbs/top5", MulticlassAccuracy(num_verbs, top_k=5, average="micro", ignore_index=-1).to(self.device), "max", "maximize")
        # self.verbs_mc = Metric("verbs/mc", MulticlassAccuracy(num_verbs, top_k=1, average="macro", ignore_index=-1).to(self.device), "max", "maximize")

        # Accuracies for nouns
        self.nouns_top1 = Metric("nouns/top1", MulticlassAccuracy(num_nouns, top_k=1, average="micro", ignore_index=-1).to(self.device), "max", "maximize")
        self.nouns_top2 = Metric("nouns/top2", MulticlassAccuracy(num_nouns, top_k=2, average="micro", ignore_index=-1).to(self.device), "max", "maximize")
        self.nouns_top3 = Metric("nouns/top3", MulticlassAccuracy(num_nouns, top_k=3, average="micro", ignore_index=-1).to(self.device), "max", "maximize")
        self.nouns_top5 = Metric("nouns/top5", MulticlassAccuracy(num_nouns, top_k=5, average="micro", ignore_index=-1).to(self.device), "max", "maximize")
        # self.nouns_mc = Metric("nouns/mc", MulticlassAccuracy(num_nouns, top_k=1, average="macro", ignore_index=-1).to(self.device), "max", "maximize")
        
        # Mean Average Precision for verbs and nouns
        self.verbs_map = Metric("verbs/map", MulticlassAveragePrecision(num_verbs, average="macro", ignore_index=-1).to(self.device), "max", "maximize")
        self.nouns_map = Metric("nouns/map", MulticlassAveragePrecision(num_nouns, average="macro", ignore_index=-1).to(self.device), "max", "maximize")
        
        # To avoid memory overflow issues (especially on A100 16GB we leave this data on the cpu)
        self.verb_logits = CatMetric().to('cpu')
        self.noun_logits = CatMetric().to('cpu')
        self.verb_labels = CatMetric().to('cpu')
        self.noun_labels = CatMetric().to('cpu')

    @torch.inference_mode()
    def update(self, outputs, losses: Tuple[torch.Tensor, ...], graphs: Data, data: Data) -> None:
        super().update(losses)
        
        verb_logits, noun_logits, _ = outputs
        verb_labels, noun_labels = data.labels
        
        verb_logits, noun_logits = verb_logits.detach().to(self.device), noun_logits.detach().to(self.device)
        verb_labels, noun_labels = verb_labels.detach().to(self.device), noun_labels.detach().to(self.device)

        # Update verb metrics
        self.verbs_top1.metric.update(verb_logits, verb_labels)
        self.verbs_top2.metric.update(verb_logits, verb_labels)
        self.verbs_top3.metric.update(verb_logits, verb_labels)
        self.verbs_top5.metric.update(verb_logits, verb_labels)
        self.verbs_map.metric.update(verb_logits, verb_labels)

        # Update noun metrics
        self.nouns_top1.metric.update(noun_logits, noun_labels)
        self.nouns_top2.metric.update(noun_logits, noun_labels)
        self.nouns_top3.metric.update(noun_logits, noun_labels)
        self.nouns_top5.metric.update(noun_logits, noun_labels)
        self.nouns_map.metric.update(noun_logits, noun_labels)
        
        # Per-class accuracies
        self.verb_logits.update(verb_logits.to('cpu'))
        self.noun_logits.update(noun_logits.to('cpu'))
        self.verb_labels.update(verb_labels.to('cpu'))
        self.noun_labels.update(noun_labels.to('cpu'))
        
    @torch.no_grad()
    def logs(self, *args) -> None: 
        verb_loss, noun_loss = self.verbs_loss.metric.compute(), self.nouns_loss.metric.compute()
        
        # # Mean Class Accuracy
        # verbs_mc = 100 * multiclass_accuracy(self.verb_logits.compute(), self.verb_labels.compute(), self.num_verbs, average='macro')
        # nouns_mc = 100 * multiclass_accuracy(self.noun_logits.compute(), self.noun_labels.compute(), self.num_nouns, average='macro')
        
        verbs_top1, verbs_top2, verbs_top3, verbs_top5 = 100 * self.verbs_top1.metric.compute(), 100 * self.verbs_top2.metric.compute(), 100 * self.verbs_top3.metric.compute(), 100 * self.verbs_top5.metric.compute()
        nouns_top1, nouns_top2, nouns_top3, nouns_top5 = 100 * self.nouns_top1.metric.compute(), 100 * self.nouns_top2.metric.compute(), 100 * self.nouns_top3.metric.compute(), 100 * self.nouns_top5.metric.compute()
        
        verbs_map, nouns_map = self.verbs_map.metric.compute(), self.nouns_map.metric.compute()
        
        logs = [
            f"Verbs: loss={verb_loss:.4f}, top_1={verbs_top1:.2f}, top_2={verbs_top2:.2f}, top_3={verbs_top3:.2f}, top_5={verbs_top5:.2f}, mAP={verbs_map:.4f}.", #, MC={verbs_mc:.2f}.",
            f"Nouns: loss={noun_loss:.4f}, top_1={nouns_top1:.2f}, top_2={nouns_top2:.2f}, top_3={nouns_top3:.2f}, top_5={nouns_top5:.2f}, mAP={nouns_map:.4f}." #, MC={nouns_mc:.2f}.",
        ]
        print_logs(logs)


    @torch.no_grad()    
    def wandb_logs(self, step: int, additional_logs: Optional[Dict[str, Any]] = None):
        super().wandb_logs(step, additional_logs)
        # verbs_class_acc = multiclass_accuracy(self.verb_logits.compute(), self.verb_labels.compute(), self.num_verbs, average='none')
        # verbs_support = torch.bincount(self.verb_labels.compute().long(), minlength=self.num_verbs)
        # nouns_class_acc = multiclass_accuracy(self.noun_logits.compute(), self.noun_labels.compute(), self.num_nouns, average='none')
        # nouns_support = torch.bincount(self.noun_labels.compute().long(), minlength=self.num_nouns)
        # verbs_acc = wandb.Table(data=[[_cls, acc, support] for _cls, (acc, support) in enumerate(zip(verbs_class_acc.numpy(), verbs_support))], columns=['label', 'acc', 'support'])
        # nouns_acc = wandb.Table(data=[[_cls, acc, support] for _cls, (acc, support) in enumerate(zip(nouns_class_acc.numpy(), nouns_support))], columns=['label', 'acc', 'support'])
        
        # # Mean Class Accuracy
        # # verbs_mc = 100 * multiclass_accuracy(self.verb_logits.compute(), self.verb_labels.compute(), self.num_verbs, average='macro')
        # # nouns_mc = 100 * multiclass_accuracy(self.noun_logits.compute(), self.noun_labels.compute(), self.num_nouns, average='macro')
        
        # wandb.log({
        #     f"{self.step_metric}": step, 
        #     f"{self.mode}/{self.prefix}/verbs/class_acc": wandb.plot.bar(verbs_acc, "label", "acc", title="Verb per-class accuracies"),  # type: ignore
        #     f"{self.mode}/{self.prefix}/nouns/class_acc": wandb.plot.bar(nouns_acc, "label", "acc", title="Noun per-class accuracies"),  # type: ignore
        #     # f"{self.mode}/{self.prefix}/verbs/mc": verbs_mc,
        #     # f"{self.mode}/{self.prefix}/nouns/mc": nouns_mc
        # })
        
    def reset(self):
        super().reset()
        self.verb_labels.reset()
        self.noun_labels.reset()
        self.verb_logits.reset()
        self.noun_logits.reset()
        