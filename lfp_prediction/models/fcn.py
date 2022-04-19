from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from lfp_prediction.models import Base


class FCN(Base):

    def __init__(self,
                 input_length: int,
                 hidden_dims: List = None,
                 output_size: int = 300,
                 **kwargs) -> None:
        super(FCN, self).__init__()

        modules = []
        if hidden_dims is None:
            hidden_dims = [512, 1024, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_length, h_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.8)
                )
            )
            input_length = h_dim

        modules.append(nn.Linear(input_length, output_size))
        self.forward_net = nn.Sequential(*modules)

    def forward(self, x, **kwargs):
        out = self.forward_net(x)
        if 'labels' in kwargs:
            labels = kwargs['labels']
            return [out, labels]
        else:
            return out

    def loss_function(self, *inputs: Any, **kwargs) -> dict:
        criterion = nn.MSELoss()
        pred = inputs[0]
        labl = inputs[1]
        l1_loss = F.l1_loss(torch.squeeze(pred), torch.squeeze(labl))
        loss = criterion(torch.squeeze(pred), torch.squeeze(labl))
        return {'loss': loss, 'rmse_loss': torch.sqrt(loss), 'l1_loss': l1_loss}
