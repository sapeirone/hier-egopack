import numpy as np
import torch

from torch_geometric.data.data import Data
from torchmetrics.aggregation import MeanMetric, CatMetric

from typing import List, Tuple, Literal, Optional

from einops import rearrange

from utils.meters.ego4d.base import BaseMeter, Metric

import editdistance
import json

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def print_logs(logs: List[str]) -> None:
    for log in logs:
        logger.info(log)
        

class TrainMeter(BaseMeter):
    
    def __init__(self, 
                 mode: Literal['train', 'val'] = 'train',
                 prefix: str = 'ego4d/lta', step_metric: str = "train/step",
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__(mode, prefix, step_metric, device)
        
        # Define loss metrics
        self.verbs_loss = Metric("verbs/loss", MeanMetric(nan_strategy='error').to(device), "min", "minimize")
        self.nouns_loss = Metric("nouns/loss", MeanMetric(nan_strategy='error').to(device), "min", "minimize")
    
    @torch.no_grad()
    def update(self, loss: Optional[Tuple[torch.Tensor, ...]]) -> None:
        # Update the loss meters
        if loss is not None:
            verbs_loss, nouns_loss = loss
            self.verbs_loss.metric.update(verbs_loss)
            self.nouns_loss.metric.update(nouns_loss)
       
    @torch.no_grad() 
    def logs(self):
        verbs_loss = self.verbs_loss.metric.compute()
        nouns_loss = self.nouns_loss.metric.compute()
        
        print_logs([f"Verbs loss={verbs_loss:.4f}. Nouns loss={nouns_loss:.4f}."])


class EvalMeter(TrainMeter):
    
    def __init__(self, Z: int, K: int,
                 mode: Literal['train', 'val'] = 'val',
                 prefix: str = 'ego4d/lta', step_metric: str = "val/step",
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__(mode, prefix, step_metric, device)
        
        self.Z = Z
        self.K = K

        # Accuracies for verbs
        self.verbs_ed = Metric("verbs/ed", MeanMetric(nan_strategy='error').to(device), "min", "minimize")
        self.nouns_ed = Metric("nouns/ed", MeanMetric(nan_strategy='error').to(device), "min", "minimize")
        
        self.clip_uids = []
        self.last_idx = CatMetric().to(device)
        self.verb_preds = CatMetric().to(device)
        self.noun_preds = CatMetric().to(device)

    @torch.no_grad()
    def update(self, outputs, losses: Tuple[torch.Tensor, ...], graphs: Data, data: Data) -> None:
        super().update(losses)
        
        verb_labels, noun_labels = data.labels
        
        verb_logits, noun_logits, _ = outputs
        verb_preds, noun_preds = generate_from_logits((verb_logits, noun_logits))
        
        # Reshape predictions and labels to match the expected format and...
        verb_preds = rearrange(verb_preds, '(b z) k -> b z k', z=self.Z, k=self.K)
        noun_preds = rearrange(noun_preds, '(b z) k -> b z k', z=self.Z, k=self.K)
        
        self.clip_uids += data.clip_uid
        self.last_idx.update(data.last_idx)
        self.verb_preds.update(verb_preds)
        self.noun_preds.update(noun_preds)
        
        if len(verb_labels) and len(noun_labels):
            verb_labels = rearrange(verb_labels, '(b z) -> b z', z=self.Z)
            noun_labels = rearrange(noun_labels, '(b z) -> b z', z=self.Z)

            # ...compute edit distance
            verbs_ed = self._edit_distance(verb_preds, verb_labels)
            nouns_ed = self._edit_distance(noun_preds, noun_labels)
            
            self.verbs_ed.metric.update(verbs_ed.to(self.device))
            self.nouns_ed.metric.update(nouns_ed.to(self.device))
        
    def _edit_distance(self, preds, labels) -> torch.Tensor:
        """
        Damerau–Levenshtein edit distance from: https://github.com/gfairchild/pyxDamerauLevenshtein.
        For each sample, we take the smallest edit distance among all the K predicted sequences.
        """
        N, Z, K = preds.shape
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        dists = []
        for n in range(N):
            dist = min([editdistance.eval(preds[n, :, k], labels[n])/Z for k in range(K)])
            dists.append(dist)
        return torch.from_numpy(np.array(dists))
        
    @torch.no_grad()
    def logs(self, *args) -> None: 
        verbs_ed, nouns_ed = self.verbs_ed.metric.compute(), self.nouns_ed.metric.compute()
        verbs_loss, nouns_loss = self.verbs_loss.metric.compute(), self.nouns_loss.metric.compute()

        print_logs([
            f"Verbs: loss={verbs_loss:.4f}, ED={verbs_ed:.4f}.",
            f"Nouns: loss={nouns_loss:.4f}, ED={nouns_ed:.4f}."
        ])
        
    def reset(self):
        super().reset()
        
        self.clip_uids = []
        self.last_idx.reset()
        self.verb_preds.reset()
        self.noun_preds.reset()
        
    def generate_inference_outputs(self):
        submissions = dict()
        
        verb_preds = self.verb_preds.compute()
        noun_preds = self.noun_preds.compute()
        
        for i, (uid, idx) in enumerate(zip(self.clip_uids, self.last_idx.compute().tolist())):
            submissions[f"{uid}_{int(idx)}"] = {
                "verb": verb_preds[i].int().T.tolist(),
                "noun": noun_preds[i].int().T.tolist()
            }

        breakpoint()
        with open('lta_w-2.json', "w", encoding="utf-8") as f_out:
            f_out.write(json.dumps(submissions))
        


def generate_from_logits(logits: Tuple[torch.Tensor, ...], K: int = 5) -> Tuple[torch.Tensor, ...]:
    """Sample from the logits.

    Parameters
    ----------
    logits : Tuple[torch.Tensor, ...]
        logits for each classification head (tensor shape: [batch_size, n_classes])
    K : int, optional
        number of samples, by default 5

    Returns
    -------
    Tuple[torch.Tensor, ...]
        the output predictions (tensor shape: [batch_size, K])
    """
    predictions: List[torch.Tensor] = []
    
    for head_logits in logits:
        preds_dist = torch.distributions.Categorical(logits=head_logits)
        preds = [preds_dist.sample() for _ in range(K)]
        head_x = torch.stack(preds, dim=1)
        predictions.append(head_x)

    return tuple(predictions)
