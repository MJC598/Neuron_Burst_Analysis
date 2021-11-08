import numpy as np
from numpy.random import default_rng
from scipy import signal, stats, io
import torch
from torch.utils.data import TensorDataset
import sys, os, random

sys.path.append(os.path.split(sys.path[0])[0])
from config import paths, params


RNG = default_rng(seed=67)

def filter_index_data(indices, lfp_raw, fr, aff, lfp_filt):

    oscBand = np.array([0.08,0.14])
    b, a = signal.butter(4,oscBand,btype='bandpass')

    data = []
    labels = []
    filt = []

    for idx in indices:
        tlfp = lfp_raw[idx-params.PREVIOUS_TIME:idx,:].reshape((-1,params.PREVIOUS_TIME*params.INPUT_FEATURES))
        # print(fr.shape)
        t_fr1 = fr[idx-params.PREVIOUS_TIME:idx,0].reshape((-1,params.PREVIOUS_TIME*params.INPUT_FEATURES))
        t_fr2 = fr[idx-params.PREVIOUS_TIME:idx,1].reshape((-1,params.PREVIOUS_TIME*params.INPUT_FEATURES))

        t_af1 = aff[idx-params.PREVIOUS_TIME:idx,0].reshape((-1,params.PREVIOUS_TIME*params.INPUT_FEATURES))
        t_af2 = aff[idx-params.PREVIOUS_TIME:idx,1].reshape((-1,params.PREVIOUS_TIME*params.INPUT_FEATURES))

        # data.append(np.concatenate((tlfp, t_fr1, t_fr2, t_af1, t_af2), axis=0))
        data.append(tlfp)
        filter_1d = signal.lfilter(b, a, lfp_raw[idx:idx+params.LOOK_AHEAD,:].reshape((-1,params.OUTPUT_SIZE)), axis=0)
        filt.append(lfp_filt[idx+params.LOOK_AHEAD,:].reshape((-1,params.OUTPUT_SIZE)))
        labels.append(filter_1d[-1,:])
    training_data = np.stack(data, axis=0)
    training_labels = np.stack(labels, axis=0)
    training_filt = np.stack(filt, axis=0).reshape((-1,1))

    return training_data, training_labels, training_filt


def get_end1D(training_samples=900000, validation_samples=100000):
    lfp_input_file = paths.RAW_LFP
    lfp_filt_file = paths.FILTERED_LFP
    fir_file = paths.FIRING_RATES
    aff_file = paths.AFFERENTS
    
    with open(lfp_input_file) as f:
        lfp_in = f.read().splitlines()
    lfp_in = np.array([float(x) for x in lfp_in]).reshape((-1, 1))
    
    with open(lfp_filt_file) as f:
        lfp_filt = f.read().splitlines()
    lfp_filt = np.array([float(x) for x in lfp_filt]).reshape((-1, 1))
    
    with open(fir_file) as f:
        fr = f.read().splitlines()
    fr = np.array([(float(x.split(',')[0]), float(x.split(',')[1])) for x in fr])
    # fr = np.array([float(x.split(',')[0]) for x in fr]).reshape((-1,1))

    with open(aff_file) as f:
        aff = f.read().splitlines()
    aff = np.array([(float(x.split(',')[0]), float(x.split(',')[1])) for x in aff])
    
    t_indices = RNG.integers(low=params.PREVIOUS_TIME, high=lfp_in.shape[0]-params.LOOK_AHEAD, size=training_samples)
    v_indices = RNG.integers(low=params.PREVIOUS_TIME, high=lfp_in.shape[0]-params.LOOK_AHEAD, size=validation_samples)
    
    train_data, train_labels, train_filt = filter_index_data(t_indices, lfp_in, fr, aff, lfp_filt)
    
    val_data, val_labels, val_filt = filter_index_data(v_indices, lfp_in, fr, aff, lfp_filt)

    # full_data, full_labels, full_filt = filter_index_data(
    #     range(params.PREVIOUS_TIME, lfp_in.shape[0]-params.LOOK_AHEAD, 1), 
    #     lfp_in, fr, lfp_filt)    
    
    print('Training Data: {}'.format(train_data.shape))
    print('Training Labels: {}'.format(train_labels.shape))
    print('Training Filter: {}'.format(train_filt.shape))
    print('Validation Data: {}'.format(val_data.shape))
    print('Validation Labels: {}'.format(val_labels.shape))
    print('Validation Filter: {}'.format(val_filt.shape))
    
    training_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_labels))
    training_filt = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_filt))
    validation_dataset = TensorDataset(torch.Tensor(val_data), torch.Tensor(val_labels))
    validation_filt = TensorDataset(torch.Tensor(val_data), torch.Tensor(val_filt))
#     f_dataset = TensorDataset(torch.Tensor(f_data), torch.Tensor(f_labels))
#     f_filt = TensorDataset(torch.Tensor(f_data), torch.Tensor(f_filt))

    return training_dataset, validation_dataset, training_filt, validation_filt#f_dataset, training_filt, validation_filt, f_filt


def get_burstLFP(in_file=paths.RAW_LFP, out_file=paths.FILTERED_LFP):
    
    fir_file = paths.FIRING_RATES
    aff_file = paths.AFFERENTS
    
    with open(in_file) as f:
        lfp_in = f.read().splitlines()
    lfp_in = np.array([float(x) for x in lfp_in]).reshape((-1, 1))
    
    with open(out_file) as f:
        lfp_out = f.read().splitlines()
    lfp_out = np.array([float(x) for x in lfp_out]).reshape((-1, 1))
    
    with open(fir_file) as f:
        fr = f.read().splitlines()
    fr = np.array([(float(x.split(',')[0]), float(x.split(',')[1])) for x in fr])
    # fr = np.array([float(x.split(',')[0]) for x in fr]).reshape((-1,1))

    with open(aff_file) as f:
        aff = f.read().splitlines()
    aff = np.array([(float(x.split(',')[0]), float(x.split(',')[1])) for x in aff])
    
    hilb = np.abs(signal.hilbert(lfp_in)) #hilbert transform of raw data

    thresh = np.mean(np.squeeze(hilb)) + 2*np.std(np.squeeze(hilb)) # 2*z_score

    indices = np.nonzero(np.squeeze(hilb)>thresh)[0]

    idx_count = params.PREVIOUS_TIME + params.LOOK_AHEAD + 1

    burst_indices = []
    temp_idx = []
    #start at the second index and compare to first one
    for i, idx in enumerate(indices[1:], 0):
        #if the index is not next start a new sample
        if idx - indices[i] != 1 and temp_idx:
            if len(temp_idx) < idx_count:
                padding = idx_count - len(temp_idx)
                pad = np.arange(temp_idx[0]-padding+1,temp_idx[0],1)
                temp_idx[:0] = pad
                if temp_idx[0] < 0:
                    temp_idx = []
                    continue
#                 print(len(temp_idx))
#             print(np.array(temp_idx).reshape((-1,1)).shape)
            burst_indices.append(np.array(temp_idx).reshape((-1,1)))
            temp_idx = []
            temp_idx.append(idx)
        #otherwise add the index to the sample
        else:
            temp_idx.append(idx)

    burst_in = []
    burst_out = []
    filt_out = []
    
    oscBand = np.array([0.08,0.14])
    b, a = signal.butter(4,oscBand,btype='bandpass')
    
    fr_t1 = fr[:,0].reshape((-1,1))
    fr_t2 = fr[:,1].reshape((-1,1))
    af_t1 = aff[:,0].reshape((-1,1))
    af_t2 = aff[:,1].reshape((-1,1))

    for sample in burst_indices:
        if sample.shape[0] >= 10:
            inp = lfp_in[np.squeeze(sample[:params.PREVIOUS_TIME])].reshape((-1,1))
            fr_i = fr_t1[np.squeeze(sample[:params.PREVIOUS_TIME])].reshape((-1,1))
            fr_p = fr_t2[np.squeeze(sample[:params.PREVIOUS_TIME])].reshape((-1,1))
            af_i = af_t1[np.squeeze(sample[:params.PREVIOUS_TIME])].reshape((-1,1))
            af_p = af_t2[np.squeeze(sample[:params.PREVIOUS_TIME])].reshape((-1,1))
            # inp = np.concatenate((inp, fr_i, fr_p, af_i, af_p), axis=1)
            filt = np.take(lfp_out, np.squeeze(sample[-1])).reshape((-1,1))
            lab = np.take(lfp_in, np.squeeze(sample[params.PREVIOUS_TIME:,:])).reshape((-1,1))
            lab = signal.lfilter(b, a, lab.reshape((-1,params.OUTPUT_SIZE)), axis=0)
            burst_in.append(inp)
            burst_out.append(lab[-1,:])
            filt_out.append(filt)
    burst_in = np.transpose(np.stack(burst_in), (0,2,1))
    burst_out = np.transpose(np.stack(burst_out).reshape((-1,1,1)), (0,2,1))
    filt_out = np.transpose(np.stack(filt_out), (0,2,1))
    print('Burst Data: {}'.format(burst_in.shape))
    print('Burst Labels: {}'.format(burst_out.shape))
    print('Filter Labels: {}'.format(filt_out.shape))
    burst_dataset = TensorDataset(torch.Tensor(burst_in), torch.Tensor(burst_out))
    filt_dataset = TensorDataset(torch.Tensor(burst_in), torch.Tensor(filt_out))
    return burst_dataset, filt_dataset


def build_invivo_data():

    oscBand = np.array([0.08,0.14])
    z, a = signal.butter(4,oscBand,btype='bandpass')

    mat = io.loadmat(paths.INVIVO_LFP)['LFP_seg']
    input_list = [] #1024 x 1 length=samples
    output_list = [] #100 x 1 length=samples
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
            # print(temp1.shape, temp2.shape)
            input_list.append(temp1)
            output_list.append(temp2)

    c = list(zip(input_list, output_list))

    random.shuffle(c)

    input_list, output_list = zip(*c)

    inputs = np.stack(input_list[:500000], axis=0)
    outputs = np.stack(output_list[:500000], axis=0)

    np.savez(paths.INVIVO_DATA, x=inputs, y=outputs)


def get_inVivo_LFP():
    if not os.path.isfile(paths.INVIVO_DATA):
        build_invivo_data()
    npfile = np.load(paths.INVIVO_DATA)
    inputs = npfile['x']
    outputs = npfile['y']
    tr_f = np.transpose(inputs[:params.TRAIN_SAMPLES, :, :], (0,2,1))
    tr_l = np.transpose(outputs[:params.TRAIN_SAMPLES, :, :], (0,2,1))
    te_f = np.transpose(inputs[params.TRAIN_SAMPLES:params.TRAIN_SAMPLES+params.VAL_SAMPLES, :, :], (0,2,1))
    te_l = np.transpose(outputs[params.TRAIN_SAMPLES:params.TRAIN_SAMPLES+params.VAL_SAMPLES, :, :], (0,2,1))
    train = TensorDataset(torch.Tensor(tr_f), torch.Tensor(tr_l))
    test = TensorDataset(torch.Tensor(te_f), torch.Tensor(te_l))
    return train, test


def get_rawLFP():
    lfp_input_file = paths.RAW_LFP
    lfp_labels_file = paths.FILTERED_LFP
    fir_file = paths.FIRING_RATES
    aff_file = paths.AFFERENTS
    with open(lfp_input_file) as f:
        lfp_in = f.read().splitlines()
    lfp_in = np.array([float(x) for x in lfp_in]).reshape((-1, 1))
    
    with open(lfp_labels_file) as f:
        lfp_out = f.read().splitlines()
    lfp_out = np.array([float(x) for x in lfp_out]).reshape((-1, 1))
#     print(lfp_out)
        
    with open(fir_file) as f:
        fr = f.read().splitlines()
#     fr = np.array([(float(x.split(',')[0]), float(x.split(',')[1])) for x in fr])
    fr = np.array([float(x.split(',')[0]) for x in fr]).reshape((-1,1))
        
    with open(aff_file) as f:
        aff = f.read().splitlines()
    aff = np.array([(float(x.split(',')[0]), float(x.split(',')[1])) for x in aff])
        
    lfp_in = lfp_in[:,0]
    fr = fr[:,0]
    # lfp_in += 1 + (-1*min(lfp_in))
    # lfp_in, _ = stats.boxcox(lfp_in)
    # lfp_in = np.diff(lfp_in, n=1, axis=0)
    # fr = np.diff(fr, n=1, axis=0)
    lfp_in = lfp_in.reshape((-1,1))
    fr = fr.reshape((-1,1))
    # full_data = np.concatenate((lfp_in, fr), axis=1)
    # print(full_data.shape)
    full_data = lfp_in
    full_labels = lfp_in
    
    training_samples = params.TRAIN_SAMPLES
    indices = RNG.integers(low=0, high=full_labels.shape[0]-(params.PREVIOUS_TIME+params.LOOK_AHEAD), size=training_samples)
    validation_samples = params.VAL_SAMPLES
    v_indices = RNG.integers(low=0, high=full_labels.shape[0]-(params.PREVIOUS_TIME+params.LOOK_AHEAD), size=validation_samples)
    training_data = []
    training_labels = []
    validation_data = []
    validation_labels = []
    f_data = []
    f_labels = []
    
    for idx in indices:
        training_data.append(full_data[idx:idx+params.PREVIOUS_TIME,:].reshape((-1,params.PREVIOUS_TIME)))
        training_labels.append(full_labels[idx+params.PREVIOUS_TIME:idx+params.PREVIOUS_TIME+params.LOOK_AHEAD,:].reshape((-1,params.OUTPUT_SIZE)))
        # training_labels.append(full_labels[idx+params.LOOK_AHEAD:idx+params.PREVIOUS_TIME+params.LOOK_AHEAD,:].reshape((-1,params.OUTPUT_SIZE)))
        # training_labels.append(full_labels[idx+params.PREVIOUS_TIME+params.LOOK_AHEAD,:].reshape((-1,params.OUTPUT_SIZE)))
    training_data = np.stack(training_data, axis=0)
    training_labels = np.stack(training_labels, axis=0)
    
    for idx in v_indices:
        validation_data.append(full_data[idx:idx+params.PREVIOUS_TIME,:].reshape((-1,params.PREVIOUS_TIME)))
        validation_labels.append(full_labels[idx+params.PREVIOUS_TIME:idx+params.PREVIOUS_TIME+params.LOOK_AHEAD,:].reshape((-1,params.OUTPUT_SIZE)))
        # validation_labels.append(full_labels[idx+params.LOOK_AHEAD:idx+params.PREVIOUS_TIME+params.LOOK_AHEAD,:].reshape((-1,params.OUTPUT_SIZE)))
        # validation_labels.append(full_labels[idx+params.PREVIOUS_TIME+params.LOOK_AHEAD,:].reshape((-1,params.OUTPUT_SIZE)))
    validation_data = np.stack(validation_data, axis=0)
    validation_labels = np.stack(validation_labels, axis=0)
    
    # for i in range(full_data.shape[0]-(params.PREVIOUS_TIME+params.LOOK_AHEAD)):
    #     f_data.append(full_data[i:i+params.PREVIOUS_TIME,:].reshape((-1,params.PREVIOUS_TIME*params.INPUT_FEATURES)))
    #     f_labels.append(full_labels[i+params.LOOK_AHEAD:i+params.PREVIOUS_TIME+params.LOOK_AHEAD,:].reshape((-1,params.OUTPUT_SIZE)))
    #     # f_labels.append(full_labels[i+params.PREVIOUS_TIME+params.LOOK_AHEAD,:].reshape((-1,params.OUTPUT_SIZE)))
    # f_data = np.stack(f_data, axis=0)
    # f_labels = np.stack(f_labels, axis=0)
    
    print('Training Data: {}'.format(training_data.shape))
    print('Training Labels: {}'.format(training_labels.shape))
    print('Validation Data: {}'.format(validation_data.shape))
    print('Validation Labels: {}'.format(validation_labels.shape))
    # print('Full Data: {}'.format(f_data.shape))
    # print('Full Labels: {}'.format(f_labels.shape))
    
    training_dataset = TensorDataset(torch.Tensor(training_data), torch.Tensor(training_labels))
    print("training_dataset built")
    validation_dataset = TensorDataset(torch.Tensor(validation_data), torch.Tensor(validation_labels))
    print("validation_dataset built")
    #f_dataset = TensorDataset(torch.Tensor(f_data), torch.Tensor(f_labels))

    return training_dataset, validation_dataset#, f_dataset


def get_filteredLFP():
    lfp_input_file = paths.FILTERED_LFP
    lfp_labels_file = paths.FILTERED_LFP
    fir_file = paths.FIRING_RATES
    aff_file = paths.AFFERENTS
    with open(lfp_input_file) as f:
        lfp_in = f.read().splitlines()
    lfp_in = np.array([float(x) for x in lfp_in]).reshape((-1, 1))
    
    with open(lfp_labels_file) as f:
        lfp_out = f.read().splitlines()
    lfp_out = np.array([float(x) for x in lfp_out]).reshape((-1, 1))
#     print(lfp_out)
        
#     with open(fir_file) as f:
#         fr = f.read().splitlines()
#     fr = np.array([(float(x.split(',')[0]), float(x.split(',')[1])) for x in fr])
        
    with open(aff_file) as f:
        aff = f.read().splitlines()
    aff = np.array([(float(x.split(',')[0]), float(x.split(',')[1])) for x in aff])
        
#     full_data = np.hstack((lfp_in, aff))
    full_data = lfp_in
    full_labels = lfp_out
    
    training_samples = 900000
    indices = RNG.integers(low=0, high=full_labels.shape[0]-(params.PREVIOUS_TIME+params.LOOK_AHEAD), size=training_samples)
    validation_samples = 100000
    v_indices = RNG.integers(low=0, high=full_labels.shape[0]-(params.PREVIOUS_TIME+params.LOOK_AHEAD), size=validation_samples)
    training_data = []
    training_labels = []
    validation_data = []
    validation_labels = []
    f_data = []
    f_labels = []
    
    for idx in indices:
        training_data.append(full_data[idx:idx+params.PREVIOUS_TIME,:].reshape((-1,params.PREVIOUS_TIME*params.INPUT_FEATURES)))
        training_labels.append(full_labels[idx+params.PREVIOUS_TIME+params.LOOK_AHEAD,:].reshape((-1,params.OUTPUT_SIZE)))
    training_data = np.stack(training_data, axis=0)
    training_labels = np.stack(training_labels, axis=0)
    
    for idx in v_indices:
        validation_data.append(full_data[idx:idx+params.PREVIOUS_TIME,:].reshape((-1,params.PREVIOUS_TIME*params.INPUT_FEATURES)))
        validation_labels.append(full_labels[idx+params.PREVIOUS_TIME+params.LOOK_AHEAD,:].reshape((-1,params.OUTPUT_SIZE)))
    validation_data = np.stack(validation_data, axis=0)
    validation_labels = np.stack(validation_labels, axis=0)
    
    for i in range(full_data.shape[0]-(params.PREVIOUS_TIME+params.LOOK_AHEAD)):
        f_data.append(full_data[i:i+params.PREVIOUS_TIME,:].reshape((-1,params.PREVIOUS_TIME*params.INPUT_FEATURES)))
        f_labels.append(full_labels[i+params.PREVIOUS_TIME+params.LOOK_AHEAD,:].reshape((-1,params.OUTPUT_SIZE)))
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


"""
get_WN -> TensorDataset

    Generates a white noise dataset from a normal distribution. Passes a 4th order
    Butterworth Filter across the data to form labels
"""
def get_WN(time_s=300000, channels=1):
    lfp_input_file = 'data/raw_data/LFP_elec_combine.txt'
    with open(lfp_input_file) as f:
        lfp_in = f.read().splitlines()
    lfp_in = np.array([float(x) for x in lfp_in]).reshape((-1, 1))
        
    noise = RNG.normal(0, np.std(lfp_in), (time_s, channels))
    
    oscBand = np.array([0.08,0.14])
    b, a = signal.butter(4,oscBand,btype='bandpass')
    noise_filt = signal.lfilter(b, a, noise, axis=0)
    
    full_data = noise
    full_labels = noise_filt
    f_data = []
    f_labels = []
    for i in range(full_data.shape[0]-(params.PREVIOUS_TIME+params.LOOK_AHEAD)):
        f_data.append(full_data[i:i+params.PREVIOUS_TIME,:].reshape((-1,params.PREVIOUS_TIME*params.INPUT_FEATURES)))
        f_labels.append(full_labels[i+params.PREVIOUS_TIME+params.LOOK_AHEAD,0].reshape((-1,params.OUTPUT_SIZE)))
    f_data = np.stack(f_data, axis=0)
    f_labels = np.stack(f_labels, axis=0)
    print('Noise Data: {}'.format(f_data.shape))
    print('Noise Labels: {}'.format(f_labels.shape))
    noise_dataset = TensorDataset(torch.Tensor(f_data), torch.Tensor(f_labels))
    return noise_dataset


def get_sin(time_s=3000, channels=1):
    A = .06 #Randomly chosen to be close to LFP magnitude
    data = []
    t_li = np.arange(0,time_s,0.001)
    for t in t_li:
        data.append(A*np.sin(t*(314.1516))) #A*np.sin((50*2*np.pi*t))
    data = np.array(data).reshape((-1,1))
    
#     oscBand = np.array([0.08,0.14])
#     b, a = signal.butter(4,oscBand,btype='bandpass')
#     data_filt = signal.lfilter(b, a, data, axis=0)
    
    full_data = data
    full_labels = data
    f_data = []
    f_labels = []
    for i in range(full_data.shape[0]-(params.PREVIOUS_TIME+params.LOOK_AHEAD)):
        f_data.append(full_data[i:i+params.PREVIOUS_TIME,:].reshape((-1,params.PREVIOUS_TIME*params.INPUT_FEATURES)))
        f_labels.append(full_labels[i+params.PREVIOUS_TIME+params.LOOK_AHEAD,:].reshape((-1,params.OUTPUT_SIZE)))
    f_data = np.stack(f_data, axis=0)
    f_labels = np.stack(f_labels, axis=0)
    print('Sin Data: {}'.format(f_data.shape))
    print('Sin Labels: {}'.format(f_labels.shape))
    sin_dataset = TensorDataset(torch.Tensor(f_data), torch.Tensor(f_labels))
    return sin_dataset