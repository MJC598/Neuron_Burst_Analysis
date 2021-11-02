import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from config import params

ROOT_DIR = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])

FILTERED_LFP = os.path.join(ROOT_DIR, 'data/raw_data/LFP_filt.txt')
# FILTERED_LFP = '/home/matt/repos/Research/Neuron_Burst_Analysis/data/raw_data/LFP_filt.txt'

RAW_LFP = os.path.join(ROOT_DIR, 'data/raw_data/LFP_elec_combine.txt')
# RAW_LFP = '/home/matt/repos/Research/Neuron_Burst_Analysis/data/raw_data/LFP_elec_combine.txt'

FIRING_RATES = os.path.join(ROOT_DIR, 'data/raw_data/FR_PN_ITN.txt')
# FIRING_RATES = '/home/matt/repos/Research/Neuron_Burst_Analysis/data/raw_data/FR_PN_ITN.txt'

AFFERENTS = os.path.join(ROOT_DIR, 'data/raw_data/AFF_PN_ITN.txt')
# AFFERENTS = '/home/matt/repos/Research/Neuron_Burst_Analysis/data/raw_data/AFF_PN_ITN.txt'

LOSS_FILE = os.path.join(ROOT_DIR, ('results/losses/bursts/losses_' + str(params.MODEL_NAME) 
                                        + '_' + params.INPUT + str(params.PREVIOUS_TIME) + '_' 
                                        + params.OUTPUT + str(params.LOOK_AHEAD) + '.csv')
                        )


PATH = os.path.join(ROOT_DIR, ('models/saved_models/LFPNet/' + str(params.MODEL_NAME) + '_' 
                                + params.INPUT + str(params.PREVIOUS_TIME) +
                                '_' + params.OUTPUT + str(params.LOOK_AHEAD) + '.pt')
                )