PREVIOUS_TIME = 300
LOOK_AHEAD = 100
INPUT_FEATURES = 1
OUTPUT_SIZE = 100  # 16
INPUT_SIZE = 1  # INPUT_FEATURES * PREVIOUS_TIME
HIDDEN_SIZE = 1  # OUTPUT_SIZE #16
BATCH_SIZE = 512
BATCH_FIRST = True
DROPOUT = 0.5
NUM_LAYERS = 2
EPOCHS = 250
TRAIN_SAMPLES = 75000
VAL_SAMPLES = 15000
# MODEL = LFPNet.LFPNetLSTM
MODEL_NAME = 'LFPNetLSTM'
OUTPUT = 'RawLFP'
INPUT = 'RawLFP'