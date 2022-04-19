import torch.nn as nn
from abc import abstractmethod
from torch import Tensor
from typing import Any
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
