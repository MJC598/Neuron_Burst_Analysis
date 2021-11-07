import scipy.io
import numpy as np
import os, sys
import random

sys.path.append(os.path.split(sys.path[0])[0])
from config import paths, params

mat = scipy.io.loadmat(paths.INVIVO_LFP)['LFP_seg']
input_list = [] #1024 x 1 length=samples
output_list = [] #100 x 1 length=samples
for arr in mat:
    # print(arr[0].shape)
    if arr[0].shape[0] < (params.PREVIOUS_TIME + params.LOOK_AHEAD + 2):
        # print(arr[0].shape[0])
        continue
    for i in range(arr[0].shape[0] - (params.PREVIOUS_TIME + params.LOOK_AHEAD + 2)):
        b = i+params.PREVIOUS_TIME
        # print(arr[0].shape)
        temp1 = np.diff(arr[0][i:b+2], n=2, axis=0)
        temp2 = np.diff(arr[0][b:b+params.LOOK_AHEAD+2], n=2, axis=0)
        input_list.append(temp1)
        output_list.append(temp2)

c = list(zip(input_list, output_list))

random.shuffle(c)

input_list, output_list = zip(*c)

inputs = np.stack(input_list[:500000], axis=0)
outputs = np.stack(output_list[:500000], axis=0)

np.savez(paths.INVIVO_DATA, x=inputs, y=outputs)
