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
        
        # self.norm = nn.BatchNorm1d(1024)
        channels = 50

        #ACTIVATION
        self.activation = nn.RRelU

        #DIALATION BRANCH (B1)
        self.dilation_branch = nn.Sequential(
            #1024 -> 512
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=3, stride=1, padding=0, dilation=257),
            self.activation,
            #512 -> 256
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=0, dilation=129),
            self.activation,
            #256 -> 128
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=62),
            self.activation
        )

        #CONVOLUTION BRANCH (B2)
        self.convolution_block1 = nn.Sequential(
            #1024 -> 512
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=7, stride=2, padding=3, dilation=1),
            self.activation,
            #512 -> 512
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=5, stride=1, padding=2, dilation=1),
            self.activation
        )

        self.convolution_block2 = nn.Sequential(
            #512 -> 256
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=5, stride=2, padding=2, dilation=1),
            self.activation,
            #256 -> 256
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1),
            self.activation,
        )

        self.convolution_block3 = nn.Sequential(
            #256 -> 128
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding= 1, dilation=1),
            self.activation,
            #128 -> 128
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=channels, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1),
            self.activation
        )

        self.pool = nn.MaxPool1d(2)

        #FCN BRANCH (B3)
        self.fcn_branch = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=1024, out_features=512),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(in_features=512, out_features=256),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(in_features=256, out_features=128),
            self.activation
        )

        self.fcn_pred_layer = nn.Sequential(
            nn.Linear(in_features=128, out_features=96),
            self.activation,
            nn.Linear(in_features=96, out_features=64),
            self.activation
        )

        #RECURRENT NETWORK
        self.rnn = nn.LSTM(input_size=in_size,hidden_size=h_size,
                           num_layers=num_layers,batch_first=params.BATCH_FIRST,dropout=dropout)


    def forward(self, x):
        # input shape: batch x feature x timestep
        # x = torch.transpose(self.norm(torch.transpose(x, 1, 2)), 1, 2)
        di_out = self.dilation_branch(x)
        c1_out = self.convolution_block1(x)
        c2_out = self.convolution_block2(c1_out + self.pool(x))
        conv_out = self.convolution_block3(c2_out + self.pool(c1_out))
        fcn_out = self.fcn_branch(x)

        # print(di_out.shape, conv_out.shape, fcn_out.shape)
        
        feature_out = di_out + conv_out + fcn_out

        pred_out = self.fcn_pred_layer(feature_out)

        pred_out = torch.transpose(pred_out, 1, 2)
        # print(feature_out.shape)
        out, (h_n, c_n) = self.rnn(pred_out)
        out = torch.squeeze(out[:,-1,:]) #self.fc(out)
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
        
        
        self.fcfinal = nn.Linear(10, 16)
        #RECURRENT NETWORK
        self.rnn = nn.LSTM(input_size=in_size,hidden_size=h_size,
                           num_layers=num_layers,batch_first=params.BATCH_FIRST,dropout=dropout)


    def forward(self, x):
        out = self.dilation128(x)
        out = out[:,:,512:]
        out = self.dilation64(out)
        out = out[:,:,-256:]
        out = self.dilation32(out)
        out = out[:,:,-128:]
        out = self.dilation16(out)
        out = out[:,:,-64:]
        out = self.dilation8(out)
        out = out[:,:,-32:]
        out = self.dilation4(out)
        out = out[:,:,-32:]
        out = self.dilation2(out)
        out = out[:,:,-16:]
        out = self.dilation1(out)
    

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