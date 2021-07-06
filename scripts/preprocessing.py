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
    thresh = np.mean(np.squeeze(hilb)) + 2*np.std(np.squeeze(hilb)) # 2*z_score
#     print(np.mean(np.squeeze(hilb)))
#     print(2*np.std(np.squeeze(hilb)))
    print(thresh)

    print(hilb.shape)
    indices = np.nonzero(np.squeeze(hilb)>thresh)[0]
    print(indices.shape)

    burst_indices = []
    temp_idx = []
    #start at the second index and compare to first one
    for i, idx in enumerate(indices[1:], 0):
        #if the index is not next start a new sample
        if idx - indices[i] != 1:
            if len(temp_idx) < 71:
                padding = 71 - len(temp_idx)
                for i in range(temp_idx[-1]+1,temp_idx[-1]+padding,1):
                    temp_idx[:0] = [i]
#             print(np.array(temp_idx).reshape((-1,1)).shape)
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
#         print(sample.shape)
        if sample.shape[0] >= 10:
            inp = np.take(lfp_in, np.squeeze(sample[:50])).reshape((-1,1))
            # print(inp.shape)
            lab = np.take(lfp_out, np.squeeze(sample[-1])).reshape((-1,1))
            # print(lab.shape)
            burst_in.append(inp)
            burst_out.append(lab)
    burst_in = np.transpose(np.stack(burst_in), (0,2,1))
    burst_out = np.transpose(np.stack(burst_out), (0,2,1))
    print(burst_in.shape)
    print(burst_out.shape)

get_burst_data()