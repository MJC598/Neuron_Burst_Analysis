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
        self.cn1 = nn.Conv1d(2, 2, kernel_size=5,padding=2)
        self.cn2 = nn.Conv1d(2, 1, kernel_size=3,padding=1)
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
        self.conv_block = nn.Sequential(
                            nn.Conv1d(2, 2, kernel_size=5, stride=1, padding=2),
                            nn.ReLU(),
                            nn.Conv1d(2, 2, kernel_size=3, stride=1, padding=1),
                            nn.ReLU()
                        )
        self.ck1s1 = nn.Conv1d(2, 1, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(int(in_size), h_size)
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

    def __init__(self, in_size, h_size, out_size, num_layers=1, dropout=0.0):
        super(LFPNetLSTM, self).__init__()

        #ACTIVATION
        self.relu = nn.ReLU()
        
        #POOLING KERNEL = 5
        self.pool = nn.MaxPool1d(5, 1)

        # STRIDE = 5
        self.shortcuts5 = nn.Conv1d(1, 1, kernel_size=1, stride=5, bias=False)
        self.convs5k3 = nn.Conv1d(1, 1, kernel_size=3, stride=5, padding=10)
        self.fcs5 = nn.Linear(50, 250)

        #STRIDE = 2
        self.shortcuts2 = nn.Conv1d(1, 1, kernel_size=1, stride=2, padding=1, bias=False)
        self.convs2k3 = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=6)
        self.fcs2 = nn.Linear(126, 250)

        #STRIDE = 1
        self.shortcuts1 = nn.Conv1d(1, 1, kernel_size=1, stride=1, bias=False)
        self.convs1k3 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=3)
        self.fcs1 = nn.Linear(250, 250)

        #RECURRENT NETWORK
        self.rnn = nn.LSTM(input_size=in_size,hidden_size=h_size,
                           num_layers=num_layers,batch_first=params.BATCH_FIRST,dropout=dropout)


    def forward(self, x):

        #S1 Branch
        residuals1 = self.shortcuts1(x)
        outs1 = self.relu(self.convs1k3(x))
        outs1 = self.pool(outs1)
        outs1 += residuals1
        outs1 = self.fcs1(outs1)

        #S2 Branch
        residuals2 = self.shortcuts2(x)
        outs2 = self.relu(self.convs2k3(x))
        outs2 = self.pool(outs2)
        outs2 += residuals2 #126 features
        outs2 = self.fcs2(outs2)

        #S5 Branch
        residuals5 = self.shortcuts5(x)
        outs5 = self.relu(self.convs5k3(x))
        outs5 = self.pool(outs5)
        outs5 += residuals5 #50 features
        outs5 = self.fcs5(outs5)

        #BRANCH CONCATENATION
        out = outs1 + outs2 + outs5      

        out = torch.transpose(out, 1, 2)

        out, (h_n, c_n) = self.rnn(out)
        out = out[:,((-1)*params.LOOK_AHEAD):,:] #self.fc(out)
        return out


class LFPNetDilatedConvLSTM(nn.Module):

    def __init__(self, in_size, h_size, out_size, num_layers=1, dropout=0.0):
        super(LFPNetDilatedConvLSTM, self).__init__()

        #ACTIVATION
        self.relu = nn.ReLU()
        
        #POOLING KERNEL = 5
        self.pool = nn.MaxPool1d(5, 1)

        # DILATIONS
        self.dilation128 = nn.Conv1d(1, 5, kernel_size=3, stride=1, dilation=128)
        self.dilation64 = nn.Conv1d(5, 5, kernel_size=3, stride=1, dilation=64)
        self.dilation32 = nn.Conv1d(5, 5, kernel_size=3, stride=1, dilation=32)
        self.dilation16 = nn.Conv1d(5, 5, kernel_size=3, stride=1, dilation=16)
        self.dilation8 = nn.Conv1d(5, 5, kernel_size=3, stride=1, dilation=8)
        self.dilation4 = nn.Conv1d(5, 5, kernel_size=3, stride=1, dilation=4)
        self.dilation2 = nn.Conv1d(5, 5, kernel_size=3, stride=1, dilation=2)
        self.dilation1 = nn.Conv1d(5, 5, kernel_size=3, stride=1, dilation=1)
        self.downchannel = nn.Conv1d(5, 1, kernel_size=1, stride=1, dilation=1, bias=False)
        self.fc = nn.Linear(1024, 1024)

        self.shortcuts5 = nn.Conv1d(1, 1, kernel_size=1, stride=5, bias=False)
        self.convs5k3 = nn.Conv1d(1, 1, kernel_size=3, stride=5, padding=10)
        self.fcs5 = nn.Linear(50, 250)

        #STRIDE = 2
        self.shortcuts2 = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=1, dilation=128, bias=False)
        self.convs2k3 = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=6)
        self.fcs2 = nn.Linear(126, 250)

        #STRIDE = 1
        self.shortcuts1 = nn.Conv1d(1, 1, kernel_size=1, stride=1, bias=False)
        self.convs1k3 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=3)
        self.fcs1 = nn.Linear(250, 250)
        
        self.fcfinal = nn.Linear(10, 16)

        #RECURRENT NETWORK
        self.rnn = nn.LSTM(input_size=in_size,hidden_size=h_size,
                           num_layers=num_layers,batch_first=params.BATCH_FIRST,dropout=dropout)


    def forward(self, x):
#         print(x.shape)
#         residual = self.shortcuts2(x)
        out = self.dilation128(x)
#         print(out.shape, residual.shape)
#         out = torch.cat((out, residual), dim=1)
#         residual = x[:,:,512:]
        out = out[:,:,512:]
        out = self.dilation64(out)
#         out = torch.cat((out, residual), dim=1)
#         residual = x[:,:,-256:]
        out = out[:,:,-256:]
        out = self.dilation32(out)
#         out = torch.cat((out, residual), dim=1)
#         residual = x[:,:,-128:]
        out = out[:,:,-128:]
        out = self.dilation16(out)
#         out = torch.cat((out, residual), dim=1)
#         residual = x[:,:,-64:]
        out = out[:,:,-64:]
        out = self.dilation8(out)
#         out = torch.cat((out, residual), dim=1)
#         residual = x[:,:,-32:]
        out = out[:,:,-32:]
        out = self.dilation4(out)
#         out = torch.cat((out, residual), dim=1)
#         residual = x[:,:,-32:]
        out = out[:,:,-32:]
        out = self.dilation2(out)
#         out = torch.cat((out, residual), dim=1)
#         residual = x[:,:,-16:]
        out = out[:,:,-16:]
        out = self.dilation1(out)
#         out = torch.cat((out, residual), dim=1)
#         out = self.downchannel(out)
    

        out = torch.transpose(out, 1, 2)
        out = out.reshape((-1,10))
#         print(out.shape)
        out = torch.unsqueeze(self.fcfinal(out),2)
#         print(out.shape)
        out, (h_n, c_n) = self.rnn(out)
        out = torch.squeeze(out[:,-1,:]) #self.fc(out)
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
        x = torch.transpose(x, 1, 2)
        # print(x.shape)
        # x, (h_n, c_n)  = self.rnn(x,(self.h0,self.c0))
        x, (h_n, c_n) = self.rnn(x)
        # print(x.shape)
        # take last cell output
        out = torch.squeeze(x[:,-1,:]) #self.lin(x[:, -1, :])

        return out


class baselineGRU(nn.Module):
    def __init__(self,input_size,hidden_size,output_size=1,
                 batch_size=1,num_layers=2,batch_first=True,dropout=0.0):
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
        out = x#self.lin(x[:, -1, :])

        return out