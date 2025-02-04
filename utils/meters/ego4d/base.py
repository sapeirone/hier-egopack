import torch

import wandb

from typing import Any, Dict, Optional, Literal
from collections import namedtuple


Metric = namedtuple("Metric", ["name", "metric", "summary", "goal"], defaults=("", None, None, None))


class BaseMeter:
    
    def __init__(self, mode: Literal['train', 'val'],
                 prefix: str, step_metric: str = "train/step", 
                 device: torch.device = torch.device("cpu")) -> None:
        self.mode = mode
        self.prefix = prefix
        self.step_metric = step_metric
        self.device = device
        
    @torch.no_grad()
    def update(self, *args, **kwargs):
        raise NotImplementedError
    
    @torch.no_grad()
    def logs(self, *args, **kwargs):
        raise NotImplementedError
   
    @torch.no_grad() 
    def define_metrics(self):
        """Iterate over all attributes of the class and define the corresponding metrics in wandb."""
        for attr_name in self.__dict__: 
            attr = getattr(self, attr_name)
            
            if isinstance(attr, Metric):
                wandb.define_metric(f"{self.mode}/{self.prefix}/{attr.name}", goal=attr.goal, summary=attr.summary, step_metric=self.step_metric)
    
    @torch.no_grad()    
    def wandb_logs(self, step: int, additional_logs: Optional[Dict[str, Any]] = None):
        logs = dict()
        for attr_name in self.__dict__: 
            attr = getattr(self, attr_name)
            
            if isinstance(attr, Metric):
                logs[f"{self.mode}/{self.prefix}/{attr.name}"] = attr.metric.compute()
        
        if additional_logs is not None:
            logs.update(additional_logs)
        
        wandb.log({f"{self.step_metric}": step, **logs})
        
    def reset(self):
        for attr_name in self.__dict__: 
            attr = getattr(self, attr_name)
            
            if isinstance(attr, Metric):
                attr.metric.reset()
