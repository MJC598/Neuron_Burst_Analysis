import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from models import LFPNet

PREVIOUS_TIME = 1024
LOOK_AHEAD = 100#16
INPUT_FEATURES = 1
OUTPUT_SIZE = 100#16
INPUT_SIZE = 1 #INPUT_FEATURES * PREVIOUS_TIME
HIDDEN_SIZE = OUTPUT_SIZE #16
BATCH_SIZE = 512
BATCH_FIRST = True
DROPOUT = 0.0
EPOCHS = 300
TRAIN_SAMPLES = 75000
VAL_SAMPLES = 15000
MODEL = LFPNet.LFPNetLSTM
MODEL_NAME = 'LFPNetLSTM'
OUTPUT = 'RawLFP'
INPUT = 'RawLFP'
RECURRENT_NET = False