from .base import Base
from .base_vae import BaseVAE
from .fcn import FCN
from .conv import CNN
from .lstm import LSTM
from .train import fit
from .beta_vae import BetaVAE
from .lstm_vae import LSTMVAE

nn_models = {
    'ConvNet': CNN,
    'FullyConnected': FCN,
    'LSTM': LSTM,
    'BetaVAE': BetaVAE,
    'LSTMVAE': LSTMVAE
}
