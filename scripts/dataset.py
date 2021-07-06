import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy import signal
import torch
from torch.utils.data import TensorDataset

rng = default_rng(seed=67)
PREVIOUS_TIME = 50
LOOK_AHEAD = 20
INPUT_FEATURES = 1
OUTPUT_SIZE = 1

def get_end1D(training_samples=900000, validation_samples=100000):
    lfp_input_file = '../data/raw_data/LFP_elec_combine.txt'
    with open(lfp_input_file) as f:
        lfp_in = f.read().splitlines()
    lfp_in = np.array([float(x) for x in lfp_in]).reshape((-1, 1))
    oscBand = np.array([0.08,0.14])
    b, a = signal.butter(4,oscBand,btype='bandpass')


    t_indices = rng.integers(low=PREVIOUS_TIME, high=lfp_in.shape[0]-LOOK_AHEAD, size=training_samples)
    v_indices = rng.integers(low=PREVIOUS_TIME, high=lfp_in.shape[0]-LOOK_AHEAD, size=validation_samples)

    training_data = []
    training_labels = []
    validation_data = []
    validation_labels = []
    f_data = []
    f_labels = []
    
    for idx in t_indices:
        training_data.append(lfp_in[idx-PREVIOUS_TIME:idx,:].reshape((-1,PREVIOUS_TIME*INPUT_FEATURES)))
        filter_1d = signal.lfilter(b, a, lfp_in[idx:idx+LOOK_AHEAD,:].reshape((-1,OUTPUT_SIZE)), axis=0)
        training_labels.append(filter_1d[-1,:])
    training_data = np.stack(training_data, axis=0)
    training_labels = np.stack(training_labels, axis=0)
    
    for idx in v_indices:
        validation_data.append(lfp_in[idx-PREVIOUS_TIME:idx,:].reshape((-1,PREVIOUS_TIME*INPUT_FEATURES)))
        filter_1d = signal.lfilter(b, a, lfp_in[idx:idx+LOOK_AHEAD,:].reshape((-1,OUTPUT_SIZE)), axis=0)
        validation_labels.append(filter_1d[-1,:])
    validation_data = np.stack(validation_data, axis=0)
    validation_labels = np.stack(validation_labels, axis=0)
    
    for i in range(PREVIOUS_TIME, lfp_in.shape[0]-LOOK_AHEAD, 1):
        f_data.append(lfp_in[i-PREVIOUS_TIME:i,:].reshape((-1,PREVIOUS_TIME*INPUT_FEATURES)))
        filter_1d = signal.lfilter(b, a, lfp_in[i:i+LOOK_AHEAD,:].reshape((-1,OUTPUT_SIZE)), axis=0)
        f_labels.append(filter_1d[-1,:])
    f_data = np.stack(f_data, axis=0)
    f_labels = np.stack(f_labels, axis=0)
    
    print('Training Data: {}'.format(training_data.shape))
    print('Training Labels: {}'.format(training_labels.shape))
    print('Validation Data: {}'.format(validation_data.shape))
    print('Validation Labels: {}'.format(validation_labels.shape))
    print('Full Data: {}'.format(f_data.shape))
    print('Full Labels: {}'.format(f_labels.shape))
    
    training_dataset = TensorDataset(torch.Tensor(training_data), torch.Tensor(training_labels))
    validation_dataset = TensorDataset(torch.Tensor(validation_data), torch.Tensor(validation_labels))
    f_dataset = TensorDataset(torch.Tensor(f_data), torch.Tensor(f_labels))

    return training_dataset, validation_dataset, f_dataset

get_end1D()