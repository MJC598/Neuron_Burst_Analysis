from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import optim

from lfp_prediction.models import Base


class FCN(Base):

    def __init__(self,
                 params: dict,
                 **kwargs) -> None:
        super(FCN, self).__init__()

        self.example_input_array = torch.empty((1, 1, 300))  # Literally just to graph the plot
        self.params = params
        self.curr_device = None

        try:
            self.input_length = self.params['input_length']
        except KeyError as e:
            raise ValueError('Parameter {} not specified'.format(e.args[0]))

        self.output_length = 300 if 'output_length' not in self.params else self.params['output_length']
        self.hidden_dims = None if 'hidden_dims' not in self.params else self.params['hidden_dims']
        self.droput = 0.8 if 'dropout' not in self.params else self.params['dropout']
        if self.hidden_dims is None:
            self.hidden_dims = [512, 1024, 512]

        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(self.input_length, h_dim),
                    nn.ReLU(),
                    nn.Dropout(p=self.droput)
                )
            )
            self.input_length = h_dim

        modules.append(nn.Linear(self.input_length, self.output_length))
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

    def training_step(self, batch, batch_idx, optimizer_idx=0, *args, **kwargs) -> STEP_OUTPUT:
        raw_signal, labels = batch
        self.curr_device = raw_signal.device

        results = self.forward(raw_signal, labels=labels)
        train_loss = self.loss_function(*results,
                                        optimizer_idx=optimizer_idx,
                                        batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        raw_signal, labels = batch
        self.curr_device = raw_signal.device

        results = self.forward(raw_signal, labels=labels)
        val_loss = self.loss_function(*results,
                                      optimizer_idx=optimizer_idx,
                                      batch_idx=batch_idx)

        self.log_dict({"val_{}".format(key): val.item() for key, val in val_loss.items()}, sync_dist=True)
        return None

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                     gamma=self.params['scheduler_gamma'])
        scheds.append(scheduler)
        return optims, scheds