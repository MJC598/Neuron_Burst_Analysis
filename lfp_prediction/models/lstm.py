import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, in_size: int = 1,
                 h_size: int = 50,
                 out_size: int = 100,
                 num_layers: int = 1,
                 batch_first: bool = True,
                 dropout: float = 0.0) -> None:
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=in_size,
                           hidden_size=h_size,
                           num_layers=num_layers,
                           batch_first=batch_first,
                           dropout=dropout)
        self.lin = nn.Linear(h_size, out_size)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)  # RNN variants expect (N, L, H)
        x, (_, _) = self.rnn(x)  # Returns (N, L, hidden_size)
        out = self.lin(x[:, -1, :])  # Feeds hidden stats of last cell to FCN

        return out
