import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from config import params


FILTERED_LFP = '/home/matt/repos/Neuron_Burst_Analysis/data/raw_data/LFP_filt.txt'
RAW_LFP = '/home/matt/repos/Neuron_Burst_Analysis/data/raw_data/LFP_elec_combine.txt'
FIRING_RATES = '/home/matt/repos/Neuron_Burst_Analysis/data/raw_data/FR_PN_ITN.txt'
AFFERENTS = '/home/matt/repos/Neuron_Burst_Analysis/data/raw_data/AFF_PN_ITN.txt'
LOSS_FILE = ('results/losses/bursts/losses_' + str(params.MODEL_NAME) + 
             '_' + params.INPUT + str(params.PREVIOUS_TIME) + '_' + params.OUTPUT + str(params.LOOK_AHEAD) + '.csv')
PATH = ('models/saved_models/LFPNet/' + str(params.MODEL_NAME) + '_' + params.INPUT + str(params.PREVIOUS_TIME) +
        '_' + params.OUTPUT + str(params.LOOK_AHEAD) + '.pt')