from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from lfp_prediction.models import Base


class CNN(Base):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 output_length: int = 300,
                 **kwargs) -> None:
        super(CNN, self).__init__()
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 64, 32, 1]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU())
                )
            in_channels = h_dim

        self.forward_net = nn.Sequential(*modules)

        self.fcfinal = nn.Linear(128, output_length)

    def forward(self, x, **kwargs):
        x = self.forward_net(x)
        out = self.fcfinal(x)
        labels = kwargs['labels']
        return [out, labels]

    def loss_function(self, *inputs: Any, **kwargs) -> dict:
        criterion = nn.MSELoss()
        pred = inputs[0]
        labl = inputs[1]
        l1_loss = F.l1_loss(torch.squeeze(pred), torch.squeeze(labl))
        loss = criterion(torch.squeeze(pred), torch.squeeze(labl))
        return {'loss': loss, 'rmse_loss': torch.sqrt(loss), 'l1_loss': l1_loss}
