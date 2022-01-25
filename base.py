import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from numpy.random import default_rng
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io
from scipy import signal
import random
import time
import pandas as pds
from sklearn.metrics import r2_score, mean_squared_error
import copy

from lfp_prediction.utils import preprocess, metrics
from lfp_prediction.config import paths, params
from lfp_prediction.models import LFPNet

s = 67

rs = RandomState(MT19937(SeedSequence(s)))
rng = default_rng(seed=s)
torch.manual_seed(s)

plt.rcParams.update({'font.size': 32})

def train_model(model,save_filepath,training_loader,validation_loader,epochs,device):
    epochs_list = []
    train_loss_list = []
    val_loss_list = []
    training_len = len(training_loader.dataset)
    validation_len = len(validation_loader.dataset)
    
#     feedback_arr = torch.zeros(params.BATCH_SIZE, 90)
    
    #splitting the dataloaders to generalize code
    data_loaders = {"train": training_loader, "val": validation_loader}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
#     loss_func = nn.L1Loss()
    decay_rate = 0.95 #decay the lr each step to 98% of previous lr
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

    total_start = time.time()

    """
    You can easily adjust the number of epochs trained here by changing the number in the range
    """
    for epoch in tqdm(range(epochs), position=0, leave=True):
        start = time.time()
        train_loss = 0.0
        val_loss = 0.0
        temp_loss = 100000000000000.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            for i, (x, y) in enumerate(data_loaders[phase]):
                if params.RECURRENT_NET:
                    x = torch.transpose(x, 2, 1)
                x = x.to(device)
                output = model(x)
                y = y.to(device)
#                 if i%100000 == 0 and epoch%5 == 0:
#                     print(output)
#                     print(y)
                loss = loss_func(torch.squeeze(output), torch.squeeze(y)) 
                #backprop             
                optimizer.zero_grad()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
#                     if i%100000 == 0 and epoch%5 == 0:
#                         print(model.cn1.weight.grad)
#                         print(model.cn2.weight.grad)
#                         print(model.fc1.weight.grad)
#                         print(model.fc2.weight.grad)

                #calculating total loss
                running_loss += loss.item()
            
            if phase == 'train':
                train_loss = running_loss
                lr_sch.step()
            else:
                val_loss = running_loss

        end = time.time()
        # shows total loss
        if epoch%5 == 0:
            tqdm.write('[%d, %5d] train loss: %.6f val loss: %.6f' % (epoch + 1, i + 1, train_loss, val_loss))
#         print(end - start)
        
        #saving best model
        if train_loss < temp_loss:
            torch.save(model, save_filepath)
            temp_loss = train_loss
        epochs_list.append(epoch)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
    total_end = time.time()
#     print(total_end - total_start)
    #Creating loss csv
    loss_df = pds.DataFrame(
        {
            'epoch': epochs_list,
            'training loss': train_loss_list,
            'validation loss': val_loss_list
        }
    )
    # Writing loss csv, change path to whatever you want to name it
    
    loss_df.to_csv(paths.LOSS_FILE, index=None)
    return train_loss_list, val_loss_list


# f_tr, f_va, f_data = preprocess.get_filteredLFP()
# f_tr, f_va, f_data = get_rawLFP()
# f_tr, f_va, t_filt, v_filt = preprocess.get_end1D()#f_data, t_filt, v_filt, f_filt = get_end1D()

# f_tr, f_va, f_data = preprocess.get_rawLFP()
f_tr, f_va = preprocess.get_raw_lfp()

# noise = get_WN(channels=2)
# sin = get_sin()

# burst, fburst = preprocess.get_burstLFP()

# Turn datasets into iterable dataloaders
train_loader = DataLoader(dataset=f_tr, batch_size=params.BATCH_SIZE)
# tfilt_loader = DataLoader(dataset=t_filt,params.BATCH_SIZE=params.BATCH_SIZE)
val_loader = DataLoader(dataset=f_va, batch_size=params.BATCH_SIZE)
# vfilt_loader = DataLoader(dataset=v_filt,params.BATCH_SIZE=params.BATCH_SIZE)

# full_loader = DataLoader(dataset=f_data,batch_size=params.BATCH_SIZE)

# ffull_loader = DataLoader(dataset=f_filt,params.BATCH_SIZE=params.BATCH_SIZE)
# noise_loader = DataLoader(dataset=noise,params.BATCH_SIZE=params.BATCH_SIZE)
# sine_loader = DataLoader(dataset=sin,params.BATCH_SIZE=params.BATCH_SIZE)

# burst_loader = DataLoader(dataset=burst,params.BATCH_SIZE=params.BATCH_SIZE)
# fburst_loader = DataLoader(dataset=fburst,params.BATCH_SIZE=params.BATCH_SIZE)

model1 = params.MODEL(params.INPUT_SIZE, params.HIDDEN_SIZE, params.OUTPUT_SIZE)
model_initial = copy.deepcopy(model1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1.to(device)

pnfr_training_loss, pnfr_validation_loss = train_model(model1, paths.PATH, train_loader,
                                                       val_loader, params.EPOCHS, device)


model1 = torch.load(paths.PATH)
model1.eval()

start = 40
k = 10000
end= (start + k) if k != None else None

model1.to('cpu')

t_pred, t_real = metrics.r2_eval(model1, train_loader, k=end)
v_pred, v_real = metrics.r2_eval(model1, val_loader, k=end)

# f_pred, f_real = r2_eval(model1, full_loader,filt=ffull_loader, k=end)
# print(f_real)
# t_pred, t_real = r2_eval(model1, train_loader, filt=None ,k=end)
# v_pred, v_real = r2_eval(model1, val_loader, filt=None, k=end)
# f_pred, f_real = r2_eval(model1, full_loader,filt=None, k=end)
# n_pred, n_real = r2_eval(model1, noise_loader, end)
# s_pred, s_real = r2_eval(model1, sine_loader, end)

# b_pred, b_real = r2_eval(model1, burst_loader, filt=fburst_loader,k=end)

# for i in range(len(s_pred)):
#     print("output: {} label: {}".format(s_pred[i], s_real[i]))

print("Train MSE: {:f}".format(mean_squared_error(t_real, t_pred)))
print("Val MSE: {:f}".format(mean_squared_error(v_real, v_pred)))
# print("Full MSE: {:f}".format(mean_squared_error(f_real, f_pred)))
# print("Burst MSE: {:f}".format(mean_squared_error(b_real, b_pred)))


# print(next(iter(train_loader))[0][:,2,0])
# print(next(iter(burst_loader))[0][:,2,0])
t_pred = t_pred[:,-1]
v_pred = v_pred[:,-1]
t_real = t_real[:,-1]
v_real = v_real[:,-1]
print(t_pred.shape)
print(t_real.shape)


fig1, ax1 = plt.subplots(nrows=1, ncols=2)
fig1.tight_layout()
ax1[0].plot(range(params.EPOCHS), pnfr_training_loss)
ax1[0].set_title('Training Loss')
ax1[0].set_ylabel('Loss')
ax1[0].set_xlabel('Epoch')

ax1[1].plot(range(params.EPOCHS), pnfr_validation_loss)
ax1[1].set_title('Validation Loss')
ax1[1].set_ylabel('Loss')
ax1[1].set_xlabel('Epoch')


fig, ax = plt.subplots(nrows=2, ncols=1)
fig.tight_layout()

ax[0].plot(np.arange(start - params.OUTPUT_SIZE, end), v_real[start - params.OUTPUT_SIZE:end], color='blue', label='Labels')
# ax[2,0].plot(np.arange(start-10,end), v_output_list[start-10:end,2], color='red',label='Internal Loop')
ax[0].scatter(np.arange(start - params.OUTPUT_SIZE, end), v_pred[start - 1:end], color='slateblue', label='Training t+10')
# ax[0].scatter(np.arange(start-params.OUTPUT_SIZE,end), v_pred[start-2:end+8,1], color='lightsteelblue',label='Training t+2')
# ax[0].scatter(np.arange(start-params.OUTPUT_SIZE,end), v_pred[start-3:end+7,2], color='gray',label='Training t+3')
# ax[0].scatter(np.arange(start-params.OUTPUT_SIZE,end), v_pred[start-4:end+6,3], color='sienna',label='Training t+4')
# ax[0].scatter(np.arange(start-params.OUTPUT_SIZE,end), v_pred[start-5:end+5,4], color='magenta',label='Training t+5')
# ax[0].scatter(np.arange(start-10,end), v_pred[start-6:end+4,5], color='aquamarine',label='Training t+6')
# ax[0].scatter(np.arange(start-10,end), v_pred[start-7:end+3,6], color='darkorange',label='Training t+7')
# ax[0].scatter(np.arange(start-10,end), v_pred[start-8:end+2,7], color='brown',label='Training t+8')
# ax[0].scatter(np.arange(start-10,end), v_pred[start-9:end+1,8], color='purple',label='Training t+9')
# ax[0].plot(np.arange(start-10,end), v_pred[start-10:end], color='green',label='Training t+10')


ax[0].set_title('Validation LFP')
ax[0].set_ylabel('LFP')
ax[0].set_xlabel('Time')
# ax[2,0].legend()

ax[1].plot(np.arange(start - params.OUTPUT_SIZE, end), t_real[start - params.OUTPUT_SIZE:end], color='blue', label='Labels')
# a[2,1].plot(np.arange(start-10,end), t_output_list[start-10:end,2], color='red',label='Internal Loop')
ax[1].scatter(np.arange(start - params.OUTPUT_SIZE, end), t_pred[start - 1:end], color='slateblue', label='Training t+10')
# ax[1].scatter(np.arange(start-params.OUTPUT_SIZE,end), t_pred[start-2:end+8,1], color='lightsteelblue',label='Training t+2')
# ax[1].scatter(np.arange(start-params.OUTPUT_SIZE,end), t_pred[start-3:end+7,2], color='gray',label='Training t+3')
# ax[1].scatter(np.arange(start-params.OUTPUT_SIZE,end), t_pred[start-4:end+6,3], color='sienna',label='Training t+4')
# ax[1].scatter(np.arange(start-params.OUTPUT_SIZE,end), t_pred[start-5:end+5,4], color='magenta',label='Training t+5')
# ax[1].scatter(np.arange(start-10,end), t_pred[start-6:end+4,5], color='aquamarine',label='Training t+6')
# ax[1].scatter(np.arange(start-10,end), t_pred[start-7:end+3,6], color='darkorange',label='Training t+7')
# ax[1].scatter(np.arange(start-10,end), t_pred[start-8:end+2,7], color='brown',label='Training t+8')
# ax[1].scatter(np.arange(start-10,end), t_pred[start-9:end+1,8], color='purple',label='Training t+9')
# ax[1].plot(np.arange(start-10,end), t_pred[start-10:end], color='green',label='Training t+10')

ax[1].set_title('Training LFP')
ax[1].set_ylabel('LFP')
ax[1].set_xlabel('Time')
ax[1].legend()

# import plotly.tools as tls
# plotly_fig = tls.mpl_to_plotly(fig)
# plotly_fig.write_html("testfile.html")
plt.show()