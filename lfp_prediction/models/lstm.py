from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import optim

from lfp_prediction.models import Base


class LSTM(Base):

    def __init__(self,
                 params: dict,
                 **kwargs) -> None:
        super(LSTM, self).__init__()
        self.save_hyperparameters(params)
        self.curr_device = None
        self.params = params
        in_features = 1 if 'in_features' not in self.params else self.params['in_features']
        hidden_size = 50 if 'hidden_size' not in self.params else self.params['hidden_size']
        out_length = 100 if 'out_length' not in self.params else self.params['out_length']
        num_lstm_layers = 1 if 'num_lstm_layers' not in self.params else self.params['num_lstm_layers']
        batch_first = True if 'batch_first' not in self.params else self.params['batch_first']
        dropout = 0.0 if 'dropout' not in self.params else self.params['dropout']
        bidirectional = False if 'bidirectional' not in self.params else self.params['bidirectional']

        self.rnn = nn.LSTM(input_size=in_features,
                           hidden_size=hidden_size,
                           num_layers=num_lstm_layers,
                           batch_first=batch_first,
                           dropout=dropout,
                           bidirectional=bidirectional)

        # lin_h_size = 2 * hidden_size if bidirectional else hidden_size
        # self.lin = nn.Linear(lin_h_size, out_length)

    def forward(self, x, **kwargs):
        x = torch.transpose(x, 1, 2)  # RNN variants expect (N, L, H)
        x, (_, _) = self.rnn(x)  # Returns (N, L, hidden_size)
        out = x[:, -1, :]
        # out = self.lin(x[:, -1, :])  # Feeds hidden stats of last cell to FCN
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
        optimizer = optim.Adam(self.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=self.params['scheduler_gamma'])
        return [optimizer], [scheduler]
