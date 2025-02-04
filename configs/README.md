# EgoPack configuration files

Configuration files are divided into two main directories `components/` and `experiments/`.

| Directory | Description |
|-----------|-------------|
| `components/` | Low-level building blocks (e.g. tasks, meters, models, etc...) |
| `experiments/` | Experiments configuration bundling components together |

## Components
Components are defined in the `components/` directory with the following structure:

| Directory       | Description                                                                                          |
|-----------------|------------------------------------------------------------------------------------------------------|
| `classifier/`   | Classifier modules (MLPs, GNNs, etc...)                                                              |
| `regressor/`    | Classifier modules (MLPs, GNNs, etc...)                                                              |
| `dataset/`      | Task-specific dataset configuration files, divided by dataset (e.g. ego4d/ar, ego4d/mq).             |
| `task/`         | Downstream tasks, divided by dataset (e.g. ego4d/ar, ego4d/mq).                                      |
| `egopack_task/` | Downstream EgoPack tasks. These tasks are supposed to inherit from the corresponding tasks in `task` |
| `model/`        | Temporal backbones (and their components).                                                           |
| `train_meter/`  | Train meters.                                                                                        |
| `eval_meter/`   | Eval meters.                                                                                         |


## Experiments
Experiments are defined in the `experiments/` directory with the following structure:

| Directory       | Description                                 |
|-----------------|---------------------------------------------|
| `single_tasks/` | Configuration files for single task models. |
| `multi_tasks/`  | Configuration files for multi task models.  |
| `egopack/`      | Configuration files for EgoPack models.     |
| `pretraining/`  | Configuration files for the pretraining.    |

Typically, these directories are organized to inherit from a common base configuration. For example, EgoPack experiments under `egopack/` extend from `egopack/_base_.yaml`.

**NOTE**: since experiments and components are defined in different directories, each configuration file defined under `experiments/` must extend the hydra's searchpath to include the `configs/components` directory. Practically, experiments file need to include the following lines at the end of the file:

```yaml
hydra:
  searchpath:
    - file://configs/experiments
    - file://configs/components
```
