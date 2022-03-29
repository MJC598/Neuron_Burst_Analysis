import os
import random

import deprecation
import numpy as np
import torch
from numpy.random import default_rng
from scipy import signal, io
from torch.utils.data import TensorDataset

from lfp_prediction.config import paths, params

RNG = default_rng(seed=67)

osc_band = np.array([0.08, 0.14])
b, a = signal.butter(4, osc_band, btype='bandpass', output='ba')


@deprecation.deprecated(deprecated_in="0.1.0", removed_in="1.0.0",
                        current_version="0.1.0",
                        details="This data is no longer expected")
def filter_index_data(indices, lfp_raw, lfp_filt):
    data = []
    labels = []
    filt = []

    for idx in indices:
        tlfp = lfp_raw[idx - params.PREVIOUS_TIME:idx, :].reshape((-1, params.PREVIOUS_TIME * params.INPUT_FEATURES))

        data.append(tlfp)
        filter_1d = signal.lfilter(b, a, lfp_raw[idx:idx + params.LOOK_AHEAD, :].reshape((-1, params.OUTPUT_SIZE)),
                                   axis=0)
        filt.append(lfp_filt[idx + params.LOOK_AHEAD, :].reshape((-1, params.OUTPUT_SIZE)))
        labels.append(filter_1d[-1, :])
    training_data = np.stack(data, axis=0)
    training_labels = np.stack(labels, axis=0)
    training_filt = np.stack(filt, axis=0).reshape((-1, 1))

    return training_data, training_labels, training_filt


@deprecation.deprecated(deprecated_in="0.1.0", removed_in="1.0.0",
                        current_version="0.1.0",
                        details="This data is no longer expected")
def get_end_1d(training_samples=900000, validation_samples=100000):
    lfp_input_file = paths.RAW_LFP
    lfp_filt_file = paths.FILTERED_LFP

    with open(lfp_input_file) as f:
        lfp_in = f.read().splitlines()
    lfp_in = np.array([float(x) for x in lfp_in]).reshape((-1, 1))

    with open(lfp_filt_file) as f:
        lfp_filt = f.read().splitlines()
    lfp_filt = np.array([float(x) for x in lfp_filt]).reshape((-1, 1))

    t_indices = RNG.integers(low=params.PREVIOUS_TIME, high=lfp_in.shape[0] - params.LOOK_AHEAD, size=training_samples)
    v_indices = RNG.integers(low=params.PREVIOUS_TIME, high=lfp_in.shape[0] - params.LOOK_AHEAD,
                             size=validation_samples)

    train_data, train_labels, train_filt = filter_index_data(t_indices, lfp_in, lfp_filt)

    val_data, val_labels, val_filt = filter_index_data(v_indices, lfp_in, lfp_filt)

    training_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_labels))
    training_filt = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_filt))
    validation_dataset = TensorDataset(torch.Tensor(val_data), torch.Tensor(val_labels))
    validation_filt = TensorDataset(torch.Tensor(val_data), torch.Tensor(val_filt))

    return training_dataset, validation_dataset, training_filt, validation_filt


@deprecation.deprecated(deprecated_in="0.1.0", removed_in="1.0.0",
                        current_version="0.1.0",
                        details="This data is no longer expected")
def get_burst_lfp(in_file=paths.RAW_LFP, out_file=paths.FILTERED_LFP):
    with open(in_file) as f:
        lfp_in = f.read().splitlines()
    lfp_in = np.array([float(x) for x in lfp_in]).reshape((-1, 1))

    with open(out_file) as f:
        lfp_out = f.read().splitlines()
    lfp_out = np.array([float(x) for x in lfp_out]).reshape((-1, 1))

    hilb = np.abs(signal.hilbert(lfp_in))  # hilbert transform of raw data

    thresh = np.mean(np.squeeze(hilb)) + 2 * np.std(np.squeeze(hilb))  # 2*z_score

    indices = np.nonzero(np.squeeze(hilb) > thresh)[0]

    idx_count = params.PREVIOUS_TIME + params.LOOK_AHEAD + 1

    burst_indices = []
    temp_idx = []
    # start at the second index and compare to first one
    for i, idx in enumerate(indices[1:], 0):
        # if the index is not next start a new sample
        if idx - indices[i] != 1 and temp_idx:
            if len(temp_idx) < idx_count:
                padding = idx_count - len(temp_idx)
                pad = np.arange(temp_idx[0] - padding + 1, temp_idx[0], 1)
                temp_idx[:0] = pad
                if temp_idx[0] < 0:
                    temp_idx = []
                    continue
            burst_indices.append(np.array(temp_idx).reshape((-1, 1)))
            temp_idx = [idx]
        # otherwise, add the index to the sample
        else:
            temp_idx.append(idx)

    burst_in = []
    burst_out = []
    filt_out = []

    for sample in burst_indices:
        if sample.shape[0] >= 10:
            inp = lfp_in[np.squeeze(sample[:params.PREVIOUS_TIME])].reshape((-1, 1))
            filt = np.take(lfp_out, np.squeeze(sample[-1])).reshape((-1, 1))
            lab = np.take(lfp_in, np.squeeze(sample[params.PREVIOUS_TIME:, :])).reshape((-1, 1))
            lab = signal.lfilter(b, a, lab.reshape((-1, params.OUTPUT_SIZE)), axis=0)
            burst_in.append(inp)
            burst_out.append(lab[-1, :])
            filt_out.append(filt)
    burst_in = np.transpose(np.stack(burst_in), (0, 2, 1))
    burst_out = np.transpose(np.stack(burst_out).reshape((-1, 1, 1)), (0, 2, 1))
    filt_out = np.transpose(np.stack(filt_out), (0, 2, 1))
    burst_dataset = TensorDataset(torch.Tensor(burst_in), torch.Tensor(burst_out))
    filt_dataset = TensorDataset(torch.Tensor(burst_in), torch.Tensor(filt_out))
    return burst_dataset, filt_dataset


@deprecation.deprecated(deprecated_in="0.1.0", removed_in="1.0.0",
                        current_version="0.1.0",
                        details="This data is no longer expected")
def build_invivo_data():
    mat = io.loadmat(paths.INVIVO_LFP)['LFP_seg']
    input_list = []  # 1024 x 1 length=samples
    output_list = []  # 100 x 1 length=samples
    full_filter_list = []  # 100 x 1 length=samples (filters with entire length then cuts down)
    for arr in mat:
        if arr[0].shape[0] < (params.PREVIOUS_TIME + params.LOOK_AHEAD):
            continue
        for i in range(arr[0].shape[0] - (params.PREVIOUS_TIME + params.LOOK_AHEAD)):
            z = i + params.PREVIOUS_TIME
            temp1 = arr[0][i:b, :]
            temp2 = signal.lfilter(b, a, arr[0][b:b + params.LOOK_AHEAD, :], axis=0)
            temp3 = (signal.lfilter(b, a, arr[0][:b + params.LOOK_AHEAD, :], axis=0))[z:, :]
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


@deprecation.deprecated(deprecated_in="0.1.0", removed_in="1.0.0",
                        current_version="0.1.0",
                        details="This data is no longer expected")
def get_invivo_lfp():
    if not os.path.isfile(paths.INVIVO_DATA):
        build_invivo_data()
    npfile = np.load(paths.INVIVO_DATA)
    inputs = npfile['x']
    outputs = npfile['y']
    full_filters = npfile['z']
    tr_f = np.transpose(inputs[:params.TRAIN_SAMPLES, :, :], (0, 2, 1))
    tr_l = np.transpose(outputs[:params.TRAIN_SAMPLES, :, :], (0, 2, 1))
    fr_f = np.transpose(full_filters[:params.TRAIN_SAMPLES, :, :], (0, 2, 1))
    te_f = np.transpose(inputs[params.TRAIN_SAMPLES:params.TRAIN_SAMPLES + params.VAL_SAMPLES, :, :], (0, 2, 1))
    te_l = np.transpose(outputs[params.TRAIN_SAMPLES:params.TRAIN_SAMPLES + params.VAL_SAMPLES, :, :], (0, 2, 1))
    fe_f = np.transpose(full_filters[params.TRAIN_SAMPLES:params.TRAIN_SAMPLES + params.VAL_SAMPLES, :, :], (0, 2, 1))
    train = TensorDataset(torch.Tensor(tr_f), torch.Tensor(tr_l))
    test = TensorDataset(torch.Tensor(te_f), torch.Tensor(te_l))
    print(fr_f.shape, fe_f.shape)
    filtered = TensorDataset(torch.Tensor(fe_f), torch.Tensor(fe_f))
    return train, test, filtered


@deprecation.deprecated(deprecated_in="0.1.0", removed_in="1.0.0",
                        current_version="0.1.0",
                        details="This data is no longer expected")
def get_raw_lfp():
    lfp_input_file = paths.RAW_LFP
    with open(lfp_input_file) as f:
        lfp_in = f.read().splitlines()
    lfp_in = np.array([float(x) for x in lfp_in]).reshape((-1, 1))

    lfp_in = lfp_in[:, 0]
    lfp_in = lfp_in.reshape((-1, 1))
    full_data = lfp_in
    full_labels = lfp_in

    training_samples = params.TRAIN_SAMPLES
    indices = RNG.integers(low=0, high=full_labels.shape[0] - (params.PREVIOUS_TIME + params.LOOK_AHEAD),
                           size=training_samples)
    validation_samples = params.VAL_SAMPLES
    v_indices = RNG.integers(low=0, high=full_labels.shape[0] - (params.PREVIOUS_TIME + params.LOOK_AHEAD),
                             size=validation_samples)
    training_data = []
    training_labels = []
    validation_data = []
    validation_labels = []

    for idx in indices:
        training_data.append(full_data[idx:idx + params.PREVIOUS_TIME, :].reshape((-1, params.PREVIOUS_TIME)))
        training_labels.append(
            full_labels[
            idx + params.PREVIOUS_TIME:idx + params.PREVIOUS_TIME + params.LOOK_AHEAD, :
            ].reshape((-1, params.OUTPUT_SIZE)))

    training_data = np.stack(training_data, axis=0)
    training_labels = np.stack(training_labels, axis=0)

    for idx in v_indices:
        validation_data.append(full_data[idx:idx + params.PREVIOUS_TIME, :].reshape((-1, params.PREVIOUS_TIME)))
        validation_labels.append(
            full_labels[
            idx + params.PREVIOUS_TIME:idx + params.PREVIOUS_TIME + params.LOOK_AHEAD, :
            ].reshape((-1, params.OUTPUT_SIZE)))
    validation_data = np.stack(validation_data, axis=0)
    validation_labels = np.stack(validation_labels, axis=0)

    training_dataset = TensorDataset(torch.Tensor(training_data), torch.Tensor(training_labels))
    validation_dataset = TensorDataset(torch.Tensor(validation_data), torch.Tensor(validation_labels))

    return training_dataset, validation_dataset


@deprecation.deprecated(deprecated_in="0.1.0", removed_in="1.0.0",
                        current_version="0.1.0",
                        details="This data is no longer expected")
def get_filtered_lfp():
    lfp_input_file = paths.FILTERED_LFP
    lfp_labels_file = paths.FILTERED_LFP
    with open(lfp_input_file) as f:
        lfp_in = f.read().splitlines()
    lfp_in = np.array([float(x) for x in lfp_in]).reshape((-1, 1))

    with open(lfp_labels_file) as f:
        lfp_out = f.read().splitlines()
    lfp_out = np.array([float(x) for x in lfp_out]).reshape((-1, 1))

    full_data = lfp_in
    full_labels = lfp_out

    training_samples = 900000
    indices = RNG.integers(low=0, high=full_labels.shape[0] - (params.PREVIOUS_TIME + params.LOOK_AHEAD),
                           size=training_samples)
    validation_samples = 100000
    v_indices = RNG.integers(low=0, high=full_labels.shape[0] - (params.PREVIOUS_TIME + params.LOOK_AHEAD),
                             size=validation_samples)
    training_data = []
    training_labels = []
    validation_data = []
    validation_labels = []
    f_data = []
    f_labels = []

    for idx in indices:
        training_data.append(
            full_data[idx:idx + params.PREVIOUS_TIME, :].reshape((-1, params.PREVIOUS_TIME * params.INPUT_FEATURES)))
        training_labels.append(
            full_labels[idx + params.PREVIOUS_TIME + params.LOOK_AHEAD, :].reshape((-1, params.OUTPUT_SIZE)))
    training_data = np.stack(training_data, axis=0)
    training_labels = np.stack(training_labels, axis=0)

    for idx in v_indices:
        validation_data.append(
            full_data[idx:idx + params.PREVIOUS_TIME, :].reshape((-1, params.PREVIOUS_TIME * params.INPUT_FEATURES)))
        validation_labels.append(
            full_labels[idx + params.PREVIOUS_TIME + params.LOOK_AHEAD, :].reshape((-1, params.OUTPUT_SIZE)))
    validation_data = np.stack(validation_data, axis=0)
    validation_labels = np.stack(validation_labels, axis=0)

    for i in range(full_data.shape[0] - (params.PREVIOUS_TIME + params.LOOK_AHEAD)):
        f_data.append(full_data[i:i + params.PREVIOUS_TIME, :].reshape((-1,
                                                                        params.PREVIOUS_TIME * params.INPUT_FEATURES)))
        f_labels.append(full_labels[i + params.PREVIOUS_TIME + params.LOOK_AHEAD, :].reshape((-1, params.OUTPUT_SIZE)))
    f_data = np.stack(f_data, axis=0)
    f_labels = np.stack(f_labels, axis=0)

    training_dataset = TensorDataset(torch.Tensor(training_data), torch.Tensor(training_labels))
    validation_dataset = TensorDataset(torch.Tensor(validation_data), torch.Tensor(validation_labels))
    f_dataset = TensorDataset(torch.Tensor(f_data), torch.Tensor(f_labels))

    return training_dataset, validation_dataset, f_dataset


"""
get_WN -> TensorDataset

    Generates a white noise dataset from a normal distribution. Passes a 4th order
    Butterworth Filter across the data to form labels
"""


@deprecation.deprecated(deprecated_in="0.1.0", removed_in="1.0.0",
                        current_version="0.1.0",
                        details="This data is no longer expected")
def get_whitenoise(time_s=300000, channels=1):
    lfp_input_file = paths.RAW_LFP
    with open(paths.RAW_LFP) as f:
        lfp_in = f.read().splitlines()
    lfp_in = np.array([float(x) for x in lfp_in]).reshape((-1, 1))

    noise = RNG.normal(0, np.std(lfp_in), (time_s, channels))

    noise_filt = signal.lfilter(b, a, noise, axis=0)

    full_data = noise
    full_labels = noise_filt
    f_data = []
    f_labels = []
    for i in range(full_data.shape[0] - (params.PREVIOUS_TIME + params.LOOK_AHEAD)):
        f_data.append(full_data[i:i + params.PREVIOUS_TIME, :].reshape((-1,
                                                                        params.PREVIOUS_TIME * params.INPUT_FEATURES)))
        f_labels.append(full_labels[i + params.PREVIOUS_TIME + params.LOOK_AHEAD, 0].reshape((-1, params.OUTPUT_SIZE)))
    f_data = np.stack(f_data, axis=0)
    f_labels = np.stack(f_labels, axis=0)
    noise_dataset = TensorDataset(torch.Tensor(f_data), torch.Tensor(f_labels))
    return noise_dataset


@deprecation.deprecated(deprecated_in="0.1.0", removed_in="1.0.0",
                        current_version="0.1.0",
                        details="This data is no longer expected")
def get_sin(time_s=3000):
    amp = .06  # Randomly chosen to be close to LFP magnitude
    data = []
    t_li = np.arange(0, time_s, 0.001)
    for t in t_li:
        data.append(amp * np.sin(t * 314.1516))  # A*np.sin((50*2*np.pi*t))
    data = np.array(data).reshape((-1, 1))

    full_data = data
    full_labels = data
    f_data = []
    f_labels = []
    for i in range(full_data.shape[0] - (params.PREVIOUS_TIME + params.LOOK_AHEAD)):
        f_data.append(full_data[i:i + params.PREVIOUS_TIME, :].reshape((-1,
                                                                        params.PREVIOUS_TIME * params.INPUT_FEATURES)))
        f_labels.append(full_labels[i + params.PREVIOUS_TIME + params.LOOK_AHEAD, :].reshape((-1, params.OUTPUT_SIZE)))
    f_data = np.stack(f_data, axis=0)
    f_labels = np.stack(f_labels, axis=0)
    sin_dataset = TensorDataset(torch.Tensor(f_data), torch.Tensor(f_labels))
    return sin_dataset
