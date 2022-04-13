import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(FCN, self).__init__()
        self.dropout = nn.Dropout(p=0.8)
        self.fc1 = nn.Linear(in_size, h_size)
        self.act = nn.ReLU()  # lambda x: x + torch.square(torch.sin(x))
        self.fc2 = nn.Linear(h_size, out_size)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        out = self.fc2(x)
        return out
    # def __init__(self, in_size, h_size, out_size):
    #     super(FCN, self).__init__()
    #     self.dropout = nn.Dropout(p=0.8)
    #     self.fc1 = nn.Linear(in_size, h_size)
    #     self.act = nn.ReLU()  # lambda x: x + torch.square(torch.sin(x))
    #     self.fc2 = nn.Linear(h_size, 750)
    #     self.fc3 = nn.Linear(750, 250)
    #     self.fc4 = nn.Linear(250, out_size)
    #
    # def forward(self, x):
    #     x = self.fc1(x)
    #     x = self.act(x)  # x + torch.square(torch.sin(x))
    #     x = self.dropout(self.fc2(x))
    #     # x = self.fc2(x)
    #     x = self.act(x)  # x + torch.square(torch.sin(x))
    #     x = self.dropout(self.fc3(x))
    #     # x = self.fc3(x)
    #     x = self.act(x)  # x + torch.square(torch.sin(x))
    #     out = self.fc4(x)
    #     return out
    