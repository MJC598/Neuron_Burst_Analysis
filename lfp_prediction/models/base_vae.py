# This model was built based on the base-vae proposed here: https://github.com/AntixK/PyTorch-VAE
# Because we wanted to modify the network for our needs, it is duplicated and changed accordingly.

import torch.nn as nn
from abc import abstractmethod
from torch import Tensor
from typing import List, Any
from lfp_prediction.models import Base


class BaseVAE(Base):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
