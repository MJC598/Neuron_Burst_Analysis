import scipy.io as io
from scipy import signal
import numpy as np
import os, sys
import random

sys.path.append(os.path.split(sys.path[0])[0])
from config import params, paths


oscBand = np.array([0.08,0.14])
z, a = signal.butter(4,oscBand,btype='bandpass')

mat = io.loadmat(paths.INVIVO_LFP)['LFP_seg']

def get_in_out():
    input_list = [] #1024 x 1 length=samples
    output_list = [] #100 x 1 length=samples
    full_filter_list = [] #100 x 1 length=samples (filters with entire length then cuts down)
    # count = 0

    for arr in mat:
        # print(arr[0].shape)
        if arr[0].shape[0] < (params.PREVIOUS_TIME + params.LOOK_AHEAD):
            # print(arr[0].shape[0])
            continue
        for i in range(arr[0].shape[0] - (params.PREVIOUS_TIME + params.LOOK_AHEAD)):
            b = i+params.PREVIOUS_TIME
            # print(arr[0].shape)
            temp1 = arr[0][i:b,:]
            temp2 = signal.lfilter(z, a, arr[0][b:b+params.LOOK_AHEAD,:], axis=0)
            temp3 = (signal.lfilter(z, a, arr[0][i:b+params.LOOK_AHEAD,:], axis=0))[-100:,:]
            # print(temp1.shape, temp2.shape, temp3.shape)
            # if count%73 == 0:
            input_list.append(temp1)
            output_list.append(temp2)
            full_filter_list.append(temp3)

            
    c = list(zip(input_list, output_list, full_filter_list))

    random.shuffle(c)

    input_list, output_list, full_filter_list = zip(*c)

    inputs = np.stack(input_list[:200000], axis=0)
    outputs = np.stack(output_list[:200000], axis=0)
    full_filters = np.stack(full_filter_list[:200000], axis=0)

    np.savez(paths.INVIVO_DATA, x=inputs, y=outputs, z=full_filters)
    # return inputs, outputs



get_in_out()