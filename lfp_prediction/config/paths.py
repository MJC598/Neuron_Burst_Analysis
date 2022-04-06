import os

ROOT_DIR = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
RESULTS_DIR = os.path.join(ROOT_DIR, '../results/')
LOSS_DIR = os.path.join(RESULTS_DIR, 'losses/')
IMG_DIR = os.path.join(RESULTS_DIR, 'images/')

MODELS_DIR = os.path.join(ROOT_DIR, 'models/saved_models/')
