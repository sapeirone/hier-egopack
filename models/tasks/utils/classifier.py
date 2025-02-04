import math

import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from typing import Tuple, Optional

import logging

logger = logging.getLogger(__name__)


class Classifier(torch.nn.Module):
    
    def __init__(self, features_size: int, n_classes: int, dropout: float = 0, prior_prob: float = 0.01, bias: bool = True, 
                 n_layers: int = 2, hidden_size: int = 512):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(features_size if i == 0 else hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.cls = nn.Linear(hidden_size, n_classes, bias=bias)
        
        if bias and prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls.bias, bias_value)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        x = self.dropout(x)
        x = self.net(x)
        return self.cls(x)


class GNNClassifier(torch.nn.Module):
    def __init__(self, features_size: int, n_classes: int, dropout: float = 0, prior_prob: float = 0.01, bias: bool = True):
        super().__init__()

        self.net = gnn.Sequential("x, edge_index, edge_weights", [
            (gnn.GraphConv(features_size, 512, aggr='max'), 'x, edge_index, edge_weights -> x'),
            (nn.LayerNorm(512)),
            (nn.ReLU(), 'x -> x'),
            (gnn.GraphConv(512, 512, aggr='max'), 'x, edge_index, edge_weights -> x'),
            (nn.LayerNorm(512)),
            (nn.ReLU(), 'x -> x'),
        ])
        self.cls = nn.Linear(512, n_classes, bias=bias)
        
        if bias and prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls.bias, bias_value)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        edge_weights = 1 / ((pos[edge_index[0]] - pos[edge_index[1]]))
        x = self.net(x, edge_index, edge_weights)
        return self.cls(x)
 

class Regressor(torch.nn.Module):
    def __init__(self, features_size: int, 
                 n_layers: int = 1, 
                 dropout: float = 0, 
                 use_ln: bool = True, 
                 use_bias: bool = True, 
                 hidden_size: Optional[int] = None):
        """Initialize an MLP regressor.

        Parameters
        ----------
        features_size : int
            number of input features
        n_layers : int, optional
            number of layers in the MLP, by default 1
        dropout : float, optional
            dropout before features projection, by default 0
        use_ln : bool, optional
            use layer normalization in the intermediate layers of the regressor, by default True
        use_bias : bool, optional
            use bias in the last linear layer of the regressor, by default True
        """
        super().__init__()
        
        hidden_size = (hidden_size or features_size)

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(*[
            self.__build_layer(features_size if i == 0 else hidden_size, hidden_size, use_ln) for i in range(n_layers)
        ])
        
        # Regression head with start and end boundaries prediction
        self.head = nn.Sequential(nn.Linear(hidden_size, 2, bias=use_bias), nn.ReLU())
        
    def __build_layer(self, features_size: int, hidden_size: int, use_ln: bool) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(features_size, hidden_size, bias=(not use_ln)), 
            nn.LayerNorm(hidden_size) if use_ln else nn.Identity(), 
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        x = self.dropout(x)
        x = self.layers(x)
        return self.head(x)
    
    
class GNNRegressor(torch.nn.Module):
    def __init__(self, features_size: int, dropout: float = 0):
        super().__init__()

        self.net = gnn.Sequential("x, edge_index, edge_weights", [
            (gnn.GraphConv(features_size, 512), 'x, edge_index, edge_weights -> x'),
            (nn.LayerNorm(512)),
            (nn.ReLU(), 'x -> x'),
            (gnn.GraphConv(512, 512), 'x, edge_index, edge_weights -> x'),
            (nn.LayerNorm(512)),
            (nn.ReLU(), 'x -> x'),
            (gnn.GraphConv(512, 2), 'x, edge_index, edge_weights -> x'),
            (nn.ReLU(), 'x -> x'),
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pos: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
        edge_weights = 1 / ((pos[edge_index[0]] - pos[edge_index[1]]))
        return self.net(x, edge_index, edge_weights)
    

class CosineClassifier(torch.nn.Module):
    def __init__(self, features_size: int, n_classes: int, dropout: float = 0):
        super().__init__()

        logger.info(f"Initializing cosine classifier module with features size {features_size} and {n_classes} classes")

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(features_size, n_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        
        x = x / x.norm(p=2, dim=1, keepdim=True)
        w = self.lin.weight / self.lin.weight.norm(p=2, dim=1, keepdim=True)
        
        return x @ w.T


class MultiHeadClassifier(torch.nn.Module):
    def __init__(self, features_size: int, n_classes: Tuple[int, ...], dropout: float = 0):
        super().__init__()

        logger.info(f"Initializing classifier module with features size {features_size} and {n_classes} classes")

        self.net = nn.ModuleList([Classifier(features_size, n_class, dropout) for n_class in n_classes])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        logits = tuple(classifier(x) for classifier in self.net)
        return logits
