from models.tasks.ego4d.ar import ARTask as Ego4dARTask
from models.tasks.ego4d.oscc import OSCCTask as Ego4dOSCCTask
from models.tasks.ego4d.lta import LTATask as Ego4dLTATask
from models.tasks.ego4d.pnr import PNRTask as Ego4dPNRTask
from models.tasks.ego4d.mq import MQTask as Ego4dMQTask

from typing import Union
TaskType = Union[Ego4dARTask, Ego4dOSCCTask, Ego4dLTATask, Ego4dPNRTask]
