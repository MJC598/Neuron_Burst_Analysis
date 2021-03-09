import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy.io
import random
import pandas as pds
import time

from scipy import stats
from sklearn.metrics import r2_score

"""
These are 3 regression RNN-based models. In order to change it to a classifier the 
nn.Linear layers must have their second parameter changed to match the number of 
expected outputs.

Expected Input Shape: (batch_size, features, time_sequence)

Input_Size - number of features
Hidden_Size - number of connections between the hidden layers
Batch_Size - How many samples you want to push through the network before executing backprop
    (this is a hyperparameter that can change how fast or slow a model converges)
Batch_First - Should always be set to True to keep input shape the same
Dropout - Only really does anything with more than 1 layer on the LSTM, RNN, GRU. Useful to help generalize training
"""

class baselineRNN(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,batch_first):
        super(baselineRNN, self).__init__()
        self.rnn1 = nn.RNN(input_size,hidden_size,batch_first=batch_first,dropout=0.5)
        self.lin = nn.Linear(hidden_size,1)
        self.h0 = torch.randn(1, batch_size, hidden_size)

    def forward(self, x):
        x, h_n  = self.rnn1(x,self.h0)

        # take last cell output
        out = self.lin(x[:, -1, :])

        return out

class baselineLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,batch_first,dropout=0.5):
        super(baselineLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size,hidden_size,batch_first=batch_first,dropout=dropout)
        self.lin = nn.Linear(hidden_size,1)
        self.h0 = torch.randn(1, batch_size, hidden_size)
        self.c0 = torch.randn(1, batch_size, hidden_size)

    def forward(self, x):
        x, (h_n, c_n)  = self.rnn(x,(self.h0,self.c0))

        # take last cell output
        out = self.lin(x[:, -1, :])

        return out

class baselineGRU(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,batch_first,dropout=0.5):
        super(baselineGRU, self).__init__()
        self.rnn = nn.GRU(input_size,hidden_size,num_layers=1,batch_first=batch_first,dropout=dropout)
        self.lin = nn.Linear(hidden_size,1)
        self.h0 = torch.randn(1, batch_size, hidden_size)

    def forward(self, x):
        # print(self.h0.shape)
        x, h_n  = self.rnn(x,self.h0)

        # take last cell output
        out = self.lin(x[:, -1, :])

        return out

def get_data_from_mat(file_path):
    data = scipy.io.loadmat(file_path)
    print(data['info_collect'].shape)
    print(data['info_collect'][0])
    duration = []
    amp = []
    pre_pn = []
    pre_itn = []
    pre_aff = []
    for i in range(1, data['info_collect'].shape[1]):
        # print(data['info_collect'][i].shape)
        duration.append(data['info_collect'][i][0])
        amp.append(data['info_collect'][i][1])
        pre_pn.append(data['info_collect'][i][2])
        pre_itn.append(data['info_collect'][i][3])
        pre_aff.append(data['info_collect'][i][4])
    duration = np.hstack(duration)
    amp = np.hstack(amp)
    pre_pn = np.hstack(pre_pn)
    pre_itn = np.hstack(pre_itn)
    pre_aff = np.hstack(pre_aff)

    #TODO

    """
    This needs to still determine the random split between training and validation along with input/output
    """
    return 0, 0

"""
Training Method

Model - Model initialized based on classes above
Save_Filepath - Where you want to save the model to. Should end with a .pt or .pth extension
    This is how you are able to load the model later for testing, etc.
training_loader - dataloader iterable with training dataset samples
validation_loader - dataloader iterable with validation dataset samples
"""

def train_model(model,save_filepath,training_loader,validation_loader):
    
    epochs_list = []
    train_loss_list = []
    val_loss_list = []
    training_len = len(training_loader.dataset)
    validation_len = len(validation_loader.dataset)

    #splitting the dataloaders to generalize code
    data_loaders = {"train": training_loader, "val": validation_loader}

    """
    This is your optimizer. It can be changed but Adam is generally used. Learning rate (alpha in gradient descent)
    is set to 0.001 but again can easily be adjusted if you are getting issues

    Loss function is set to Mean Squared Error. If you switch to a classifier I'd recommend switching the loss function to
    nn.CrossEntropyLoss(), but this is also something that can be changed if you feel a better loss function would work
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()

    total_start = time.time()

    """
    You can easily adjust the number of epochs trained here by changing the number in the range
    """
    for epoch in range(20):
        start = time.time()
        train_loss = 0.0
        val_loss = 0.0
        temp_loss = 10000.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            for i, (x, y) in enumerate(data_loaders[phase]):  
                # This permutation is done so it fits into the model. I reversed the features and sequence length with my EEG data
                # So I had to fix it here, it will come back later with another permute
                x = x.permute(0, 2, 1)
                output = model(x)
                #Computing loss                              
                loss = loss_func(torch.squeeze(output), torch.squeeze(y))  
                #backprop             
                optimizer.zero_grad()           
                if phase == 'train':
                    loss.backward()
                    optimizer.step()                                      

                #calculating total loss
                running_loss += loss.item()
            
            if phase == 'train':
                train_loss = running_loss
            else:
                val_loss = running_loss

        end = time.time()
        # shows average loss
        # print('[%d, %5d] train loss: %.6f val loss: %.6f' % (epoch + 1, i + 1, train_loss/training_len, val_loss/validation_len))
        # shows total loss
        print('[%d, %5d] train loss: %.6f val loss: %.6f' % (epoch + 1, i + 1, train_loss, val_loss))
        print(end - start)
        
        #saving best model
        if val_loss < temp_loss:
            torch.save(model, save_filepath)
            temp_loss = val_loss
        epochs_list.append(epoch)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
    total_end = time.time()
    print(total_end - total_start)
    #Creating loss csv
    loss_df = pds.DataFrame(
        {
            'epoch': epochs_list,
            'training loss': train_loss_list,
            'validation loss': val_loss_list
        }
    )
    # Writing loss csv, change path to whatever you want to name it
    loss_df.to_csv('losses_csv.csv', index=None)

"""
General R2 Scoring
Model - same model as sent to train_model
testing_dataloader - whichever dataloader you want to R2 Score
"""
def r2_score_eval(model, testing_dataloader):
    output_list = []
    labels_list = []
    for i, (x, y) in enumerate(testing_dataloader):      
        # Same permute issue we had in training. Basically switching from (batch_size, features, time) 
        # to (batch_size, time, features) 
        x = x.permute(0, 2, 1)
        output = model(x) 
        output_list.append(np.squeeze(np.transpose(output.detach().cpu().numpy())))
        labels_list.append(y.detach().cpu().numpy())
    output_list = np.hstack(output_list)
    labels_list = np.hstack(labels_list)
    print(r2_score(labels_list, output_list))

if __name__ == "__main__":
    input_size = 25
    hidden_size = 251
    batch_first = True
    batch_size = 32
    model = baselineLSTM(input_size,hidden_size,batch_size,batch_first,0)
    # model = baselineGRU(input_size,hidden_size,batch_size,batch_first,0)
    # model = baselineRNN(input_size,hidden_size,batch_size,batch_first)
    training_dataset, validation_dataset = get_data_from_mat('info_collect_for_NN_network.mat') #retrieve data function

    # Turn datasets into iterable dataloaders
    training_loader = DataLoader(dataset=training_dataset,batch_size=batch_size,shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset,batch_size=batch_size)

    PATH = 'baselineLSTM.pth'
    train_model(model,PATH,training_loader,validation_loader)
    model = torch.load(PATH)
    model.eval()
    r2_score_eval(model, training_loader)
    r2_score_eval(model, validation_loader)