<div align="center">
<img align="left" height="75" style="margin-left: 20px" src="assets/logo.png" alt="">



# Hier-EgoPack: Hierarchical Egocentric Video Understanding with Diverse Task Perspectives

[Simone Alberto Peirone](https://scholar.google.com/citations?user=K0efPssAAAAJ), [Francesca Pistilli](https://scholar.google.com/citations?user=7MJdvzYAAAAJ), [Antonio Alliegro](https://scholar.google.com/citations?user=yQqW5q0AAAAJ), [Tatiana Tommasi](https://scholar.google.com/citations?user=ykFtI-QAAAAJ), [Giuseppe Averta](https://scholar.google.com/citations?user=i4rm0tYAAAAJ)

</div>

<div align="center">

<a href='https://arxiv.org/abs/2502.02487' style="margin: 10px"><img src='https://img.shields.io/badge/Paper-Arxiv:2502.02487-red'></a>
<a href='https://sapeirone.github.io/hier-egopack/' style="margin: 10px"><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a target="_blank" href="https://colab.research.google.com/github/sapeirone/hier-egopack/blob/main/quickstart.ipynb" style="margin: 10px">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

✨ <strong>This paper extends our previous work <a href="https://sapeirone.github.io/EgoPack/">A Backpack Full of Skills: Egocentric Video Understanding with Diverse Task Perspectives" (CVPR 2024)</a></strong> ✨
</div>
<br>

**Abstract:** 

Our comprehension of video streams depicting human activities is naturally multifaceted: in just a few moments, we can grasp what is happening, identify the relevance and interactions of objects in the scene, and forecast what will happen soon, everything all at once. To endow autonomous systems with such a holistic perception, learning how to correlate concepts, abstract knowledge across diverse tasks, and leverage tasks synergies when learning novel skills is essential.
A significant step in this direction is EgoPack, a unified framework for understanding human activities across diverse tasks with minimal overhead. EgoPack promotes information sharing and collaboration among downstream tasks, essential for efficiently learning new skills.

In this paper, we introduce Hier-Egopack, which advances EgoPack by enabling reasoning also across diverse temporal granularities, which expands its applicability to a broader range of downstream tasks. 
To achieve this, we propose a novel hierarchical architecture for temporal reasoning equipped with a GNN layer specifically designed to tackle the challenges of multi-granularity reasoning effectively.
We evaluate our approach on multiple Ego4d benchmarks involving both clip-level and frame-level reasoning, demonstrating how our hierarchical unified architecture effectively solves these diverse tasks simultaneously.


## Table of Contents

- [Hier-EgoPack: Hierarchical Egocentric Video Understanding with Diverse Task Perspectives](#hier-egopack-hierarchical-egocentric-video-understanding-with-diverse-task-perspectives)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Environment setup](#environment-setup)
      - [Additional dependencies](#additional-dependencies)
  - [Data](#data)
  - [Training and Evaluation](#training-and-evaluation)
    - [Single Task Training](#single-task-training)
    - [Multi Task Training](#multi-task-training)
    - [EgoPack](#egopack)
  - [Adding a novel task with Hier-EgoPack](#adding-a-novel-task-with-hier-egopack)
    - [Step 1: Dataset definition](#step-1-dataset-definition)
    - [Step 2: Task definition](#step-2-task-definition)
  - [Acknowledgements](#acknowledgements)
  - [Cite Us](#cite-us)



## Getting Started

```
git clone git@github.com:sapeirone/hier-egopack.git hier-egopack
cd hier-egopack
mkdir checkpoints
```

### Environment setup

```
conda create --name hier-egopack python=3.10
conda activate hier-egopack
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install torch_geometric
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip3 install -r requirements.txt
```

#### Additional dependencies
This work uses the NMS implementation from [ActionFormer](https://github.com/happyharrycn/actionformer_release) which needs to be manually compiled.

```
cd libs/utils/
python setup.py install --user
cd ../..
```

## Data
This work is built on several benchmarks from Ego4d. Download the v1 annotations and the **omnivore_video_swinl** features following the [official docs](https://ego4d-data.org/docs/CLI/). Download our pre-extracted **egovlp** features from [here]().

```
mkdir -p data/ego4d/raw/
ln -s /path/to/ego4d/annotations $(pwd)/data/ego4d/raw/
ln -s /path/to/ego4d/features $(pwd)/data/ego4d/raw/
```

This should result in the following directory structure under `data/ego4d/raw/`:
```
.
└─── annotations
     └─── v1
     │    │ ego4d.json
     │    │ ...
     │
     └─── features
          └─── omnivore_video_swinl
          │    │ 64b355f3-ef49-4990-8622-9e9eef68b495.pth
          │    │ ...
          │     
          └─── egovlp
               │ 64b355f3-ef49-4990-8622-9e9eef68b495.pth
               │ ...
```



## Training and Evaluation

### Single Task Training

To train single task models, use the following commands:

| Task                                          | Command                                          |
|-----------------------------------------------|--------------------------------------------------|
| Action Recognition (**AR**)                   | `python train_single_task.py --config-name=ar name_pattern='\{task.name\}' save_to=path/to/models/`   |
| Object State Change Classification (**OSCC**) | `python train_single_task.py --config-name=oscc name_pattern='\{task.name\}' save_to=path/to/models/` |
| Long Term Anticipation (**LTA**)              | `python train_single_task.py --config-name=lta name_pattern='\{task.name\}' save_to=path/to/models/`  |
| Moment Queries (**MQ**)                       | `python train_single_task.py --config-name=mq name_pattern='\{task.name\}' save_to=path/to/models/`   |
| Point of No Return (**PNR**)                  | `python train_single_task.py --config-name=pnr name_pattern='\{task.name\}' save_to=path/to/models/`  |

These training generate a checkpoint file under `path/to/models/` with name `{task.name}_YYYY-MM-DD-HH-mm.pth`.

By default, models are trained using **egovlp** features. To train models using omnivore video features, add `features=ego4d/omnivore_video` to the command.

### Multi Task Training

```bash
python train_multi_task.py --config-name=multi_task name_pattern={mtl_tasks} mtl_tasks=[ego4d/ar,ego4d/lta,ego4d/mq,ego4d/oscc,ego4d/pnr] save_to=path/to/models/`
```
This training generate a checkpoint file under `path/to/models/` with name `{mtl_tasks}_YYYY-MM-DD-HH-mm.pth`, with '/' replaced by '_'. 

**Example:** `path/to/models/ego4d_ar-ego4d_lta-ego4d_oscc-ego4d_pnr_2024-11-09-13-36.pth`.

### EgoPack

| Task                                          | Command                                          |
|-----------------------------------------------|--------------------------------------------------|
| Action Recognition (**AR**)                   | `python train_egopack.py --config-name=ar name_pattern='\{task.name\}' resume_from=path/to/mtl_models/`   |
| Object State Change Classification (**OSCC**) | `python train_egopack.py --config-name=oscc name_pattern='\{task.name\}' resume_from=path/to/mtl/models/` |
| Long Term Anticipation (**LTA**)              | `python train_egopack.py --config-name=lta name_pattern='\{task.name\}' resume_from=path/to/mtl/models/`  |
| Moment Queries (**MQ**)                       | `python train_egopack.py --config-name=mq name_pattern='\{task.name\}' resume_from=path/to/mtl/models/`   |
| Point of No Return (**PNR**)                  | `python train_egopack.py --config-name=pnr name_pattern='\{task.name\}' resume_from=path/to/mtl/models/`  |

## Adding a novel task with Hier-EgoPack
Adding a novel tasks requires creating a few files that define the dataset, models and metrics associated to the task. Here, we report a step-by-step guide to adding a novel task, e.g. Moment Queries, to the Hier-EgoPack framework.

### Step 1: Dataset definition
Add the dataset file, e.g. `data/ego4d/novel.py`, and the corresponding config file, e.g. `configs/components/dataset/ego4d/novel.yaml`. The `_target_` parameter in the config file should point to the dataset class. 

Config file example:
```yaml
_target_: data.ego4d.novel.NovelDataset
version: 1  # ego4d annotations version

root: data/ego4d  # ego4d data root 
```

The dataset class should return a torch geometric `Data` object representing the video sample as a graph, with its associated metadata:
- `video_uid` and `clip_uid` represent the video and clip associated to the input (as per the official ego4d terminology),
- `x` are the features of the graph nodes,
- `indices` is a integer index to identify nodes in the graph,
- `pos` represents the timestamp associated to each node.
- `y` is the label associated to the graph (e.g., for OSCC a binary label that indicates the presence of an object state change) or a set of node-wise labels. **Depending on the task, one could return the ground truth in different ways.** For examples, the MomentQueries dataset returns the list of labels and respective time windows. The task and meter classes should be adapted accordingly to support the selected format.

```python
import torch
from data.ego4d.Ego4dDataset import BaseDataset
from torch_geometric.data import Data

class NovelDataset(BaseDataset):

    def __getitem__(self, idx: int) -> Data:
        node_features = torch.randn((N, 1, 256))
        features_stride, input_fps = 16, 30

        return Data(
            video_uid='xyz',
            clip_uid='xyz',
            x=node_features,
            mask=torch.ones((N, )).bool(),
            indices=torch.arange(len(feats)),
            pos=(0.5 + torch.arange(len(feats))) * features_stride / input_fps,
            y=...,
            ...
        )

```

### Step 2: Task definition
You need to define two classes, a **task** and an **EgoPack task**. 

The task class is used in single task and multi-task training scenario. It takes the output of the temporal backbone (a set of graphs at different temporal resolutions), maps it to the features space of the task (`self.project(...)` transform) and finally to the output space of the task.

A simple skeleton of a task class looks as follows:
```python
import torch.nn as nn
from torch_geometric.data import Data
from typing import Any

from models.tasks.task import Task, EgoPackTask

class NovelTask(Task):

  def __init__(self, name: str, input_size: int, features_size: int, N: int):
      super().__init__(name, input_size, features_size, dropout)

      # define the task components here
      self.classifier = nn.Linear(features_size, N)
          
  def forward(self, graphs: Data, data: Data, **kwargs) -> torch.Tensor:
      features: Tensor = graphs.x

      # graphs contains the output graphs from the temporal model
      # data is the original graph from the dataset

      # task-specific projection
      features = self.project(features)

      return self.classifier(features)

  def compute_loss(self, outputs: torch.Tensor, graphs: Data, data: Data):

      return loss_fn(outputs, data.y)
```

The EgoPack task class extend the base task by fusing the output of the novel task with the feature of the support tasks. Specifically, its `forward` method projects the nodes in the features space of the novel task and then combines (averages) these features with the ones of the support tasks.

Example:
```python
from models.tasks.task import Task, EgoPackTask

class EgoPackNovelTask(NovelTask, EgoPackTask):

def __init__(self, *args, **kwargs):
    super().__init__(**kwargs)
        
# pylint: disable=unused-argument,arguments-differ
def forward(self, graphs: Data, data: Data, auxiliary_features: Dict[str, Tensor], **kwargs) -> Tuple[Tensor, ...]:
    features: Tensor = graphs.x
    
    # task-specific projection
    features = self.project(features)

    # features fusion
    features = combine(features, auxiliary_features)
    
    return self.classifier(features)
```
 
## Acknowledgements
This study was carried out within the FAIR - Future Artificial Intelligence Research and received funding from the European Union Next-GenerationEU (PIANO NAZIONALE DI RIPRESA E RESILIENZA (PNRR) – MISSIONE 4 COMPONENTE 2, INVESTIMENTO 1.3 – D.D. 1555 11/10/2022, PE00000013). This manuscript reflects only the authors’ views and opinions, neither the European Union nor the European Commission can be considered responsible for them. We acknowledge the CINECA award under the ISCRA initiative, for the availability of high performance computing resources and support. Antonio Alliegro and Tatiana Tommasi also acknowledge the EU project ELSA - European Lighthouse on Secure and Safe AI (grant number 101070617).


## Cite Us

```
@article{peirone2025backpack,
  title   = {Hier-EgoPack: Hierarchical Egocentric Video Understanding with Diverse Task Perspectives},
  author  = {Peirone, Simone Alberto and Pistilli, Francesca and Alliegro, Antonio and Tommasi, Tatiana and Averta, Giuseppe},
  journal = {arXiv preprint arXiv:2502.},
  year    = {2025}
}
```


Please consider also citing our original CVPR publication:
```
@inproceedings{peirone2024backpack,
    title     = {A Backpack Full of Skills: Egocentric Video Understanding with Diverse Task Perspectives}, 
    author    = {Simone Alberto Peirone and Francesca Pistilli and Antonio Alliegro and Giuseppe Averta},
    year      = {2024},
    booktitle = {Proceedings of the IEEE/CVF conference on computer vision and pattern recognition}
}
```