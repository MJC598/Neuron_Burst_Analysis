import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(CNN, self).__init__()
        self.bn1 = nn.BatchNorm1d(1)
        self.bn6 = nn.BatchNorm1d(6)
        self.bn18 = nn.BatchNorm1d(18)
        self.convstride8 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=7, stride=8, dilation=1)

        self.convstride16 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=7, stride=16, dilation=1)

        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.res_block = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ReLU()
        )

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=1, dilation=1)  # 2044

        self.conv2 = nn.Conv1d(in_channels=6, out_channels=6, kernel_size=5, stride=1, dilation=1)

        self.convfm = nn.Conv1d(in_channels=6, out_channels=1, kernel_size=1, stride=1, dilation=1)

        self.dropout = nn.Dropout(p=0.8)
        self.act = nn.ReLU()
        self.fcfinal = nn.Linear(128, out_size)

    def forward(self, x):
        x = self.bn1(x)
        # y1 = self.act(self.convstride16(self.dropout(x)))  # batch x 6 x 128

        # y2 = self.act(self.convstride8(self.dropout(x)))  # batch x 6 x 256

        # y2 = self.pool(self.res_block(y2))  # batch x 6 x 128

        y3 = self.act(self.conv1(self.dropout(x)))  # 2044
        res = y3
        y3 = self.res_block(self.bn6(y3))
        y3 += res
        y3 = self.pool(y3)  # 1022
        y3 = self.act(self.conv2(self.dropout(y3)))  # 1018
        res = y3
        y3 = self.res_block(self.bn6(y3))
        y3 += res
        y3 = self.pool(y3)  # 509
        y3 = self.pool(y3)  # 255
        y3 = self.pool(y3)  # 128

        # y = self.bn18(torch.cat((y1, y2, y3), dim=1))
        y = y3

        y = self.act(self.convfm(y))
        out = self.fcfinal(y)
        return out
