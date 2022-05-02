import torch.nn as nn
from abc import abstractmethod

from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from typing import Any, Optional
import pytorch_lightning as pl


class Base(pl.LightningModule):
    def __init__(self) -> None:
        super(Base, self).__init__()

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        pass

    @abstractmethod
    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass
