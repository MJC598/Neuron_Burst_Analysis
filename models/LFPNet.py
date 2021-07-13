import torch
import torch.nn as nn
import os, sys

from torch.nn.modules import padding
sys.path.append(os.path.split(sys.path[0])[0])

from config import params


class FCN(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(FCN,self).__init__()
        self.fc1 = nn.Linear(in_size, h_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(h_size, out_size)
    def forward(self, x):
        x = self.tanh(self.fc1(x))
        out = self.fc2(x)
        return out


#Formerly Conv1dFCN
class LFPNet1C(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(LFPNet1C, self).__init__()
        self.cn1 = nn.Conv1d(1, 1, kernel_size=5,padding=2)
        self.cn2 = nn.Conv1d(1, 1, kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(int(in_size), h_size)
        self.fc2 = nn.Linear(h_size, out_size)
    
    def forward(self,x):
        residual = x
        x = self.relu(self.cn1(x))
        x = self.relu(self.cn2(x))
        x += residual
        
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out


class LFPNetMC(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(LFPNetMC, self).__init__()
        self.ck5s3 = nn.Conv1d(1, 1, kernel_size=5, stride=3)
        self.ck3s3 = nn.Conv1d(1, 1, kernel_size=3, stride=3)
        self.conv_block = nn.Sequential(
                            nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2),
                            nn.ReLU(),
                            nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1),
                            nn.ReLU()
                        )
        self.ck1s1 = nn.Conv1d(1, 1, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(int(in_size), h_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(h_size, out_size)
    
    def forward(self,x):
        residual = x
        x = self.conv_block(x)
        x += residual
        
        residual = x
        x = self.conv_block(x)
        x += residual
        
        x = self.relu(self.ck1s1(x))
        
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out


class LFPNetLSTM(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(LFPNetLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=params.LOOK_AHEAD, batch_first=params.BATCH_FIRST)
        self.fc = nn.Linear(params.PREVIOUS_TIME, out_size)
        self.conv_s5_block = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5, stride=5, padding=0),
            nn.ReLU(),
            nn.Conv1d(1, 1, kernel_size=3, stride=5, padding=0),
            nn.ReLU(),
            nn.Linear(4, params.PREVIOUS_TIME)
        )
        self.conv_s3_block = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(1, 1, kernel_size=3, stride=3, padding=2),
            nn.ReLU(),
            nn.Linear(12, params.PREVIOUS_TIME)
        )
        self.conv_s1_block = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Linear(params.PREVIOUS_TIME, params.PREVIOUS_TIME)
        )
        self.conv = nn.Conv1d(3, 1, kernel_size=1, stride=1)

    def forward(self, x):
        o1 = self.conv_s5_block(x)
        o2 = self.conv_s3_block(x)
        o3 = self.conv_s1_block(x)
        o = torch.cat((o1, o2, o3), 1)
        out = self.conv(o)
        # print(out.size())
        out = self.fc(out)
        # out = self.fc(self.lstm(out))
        # print(out.size())
        return out


class baselineRNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size=1,
                 batch_size=1,num_layers=1,batch_first=True,dropout=0.0):
        super(baselineRNN, self).__init__()
        self.rnn1 = nn.RNN(input_size=input_size,hidden_size=hidden_size,
                           num_layers=num_layers,batch_first=batch_first,dropout=dropout)
        self.lin = nn.Linear(hidden_size,output_size)
        # self.h0 = torch.randn(num_layers, batch_size, hidden_size)

    def forward(self, x):
        # x, h_n  = self.rnn1(x,self.h0)
        x, h_n  = self.rnn1(x)

        # take last cell output
        out = self.lin(x[:, -1, :])

        return out


class baselineLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size=1,
                 batch_size=1,num_layers=1,batch_first=True,dropout=0.0):
        super(baselineLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,
                           num_layers=num_layers,batch_first=batch_first,dropout=dropout)
        self.lin = nn.Linear(hidden_size,output_size)
        # self.h0 = torch.randn(num_layers, batch_size, hidden_size)
        # self.c0 = torch.randn(num_layers, batch_size, hidden_size)

    def forward(self, x):
        # x, (h_n, c_n)  = self.rnn(x,(self.h0,self.c0))
        x, (h_n, c_n) = self.rnn(x)
        # take last cell output
        out = self.lin(x[:, -1, :])

        return out


class baselineGRU(nn.Module):
    def __init__(self,input_size,hidden_size,output_size=1,
                 batch_size=1,num_layers=1,batch_first=True,dropout=0.0):
        super(baselineGRU, self).__init__()
        self.rnn = nn.GRU(input_size=input_size,hidden_size=hidden_size,
                          num_layers=num_layers,batch_first=batch_first,dropout=dropout)
        self.lin = nn.Linear(hidden_size,output_size)
        # self.h0 = torch.randn(num_layers, batch_size, hidden_size)

    def forward(self, x):
        # print(self.h0.shape)
        # x, h_n  = self.rnn(x,self.h0)
        x, h_n = self.rnn(x)

        # take last cell output
        out = self.lin(x[:, -1, :])

        return out