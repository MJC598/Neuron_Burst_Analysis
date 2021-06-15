import numpy as np
from scipy import stats, signal
from numpy.random import default_rng
import sys
np.set_printoptions(threshold=sys.maxsize)

DEFAULT_RAW_FILE = '../data/raw_data/LFP_elec_combine.txt'
DEFAULT_LABELS_FILE = '../data/raw_data/LFP_filt.txt'

def get_burst_data(in_file=DEFAULT_RAW_FILE, out_file=DEFAULT_LABELS_FILE):
    with open(in_file) as f:
        lfp_in = f.read().splitlines()
    lfp_in = np.array([float(x) for x in lfp_in]).reshape((-1, 1))
    
    with open(out_file) as f:
        lfp_out = f.read().splitlines()
    lfp_out = np.array([float(x) for x in lfp_out]).reshape((-1, 1))
    
    hilb = np.abs(signal.hilbert(lfp_out)) #hilbert transform of raw data

    z_score = stats.zscore(hilb)
    thresh = 2*z_score #mean of hilb + 2*stddev of hilb
    print(thresh.shape)

    indices = np.transpose((np.squeeze(hilb)>np.squeeze(thresh)).nonzero())

    burst_indices = []
    temp_idx = []
    #start at the second index and compare to first one
    for i, idx in enumerate(indices[1:,0], 0):
        #if the index is not next start a new sample
        if idx - indices[i, 0] != 1:
            burst_indices.append(np.array(temp_idx).reshape((-1,1)))
            temp_idx = []
            temp_idx.append(idx)
        #otherwise add the index to the sample
        else:
            temp_idx.append(idx)

    # print(np.squeeze(indices))
    burst_in = []
    burst_out = []
    for sample in burst_indices:
        if sample.shape[0] > 10:
            # print(sample.shape)
            burst_in.append(np.take(lfp_in, np.squeeze(sample)))
            burst_out.append(np.take(lfp_out, np.squeeze(sample)))
    print(len(burst_in))
    print(len(burst_out))

get_burst_data()