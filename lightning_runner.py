import argparse
import os
from pathlib import Path

import yaml

from lfp_prediction.models import nn_models
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lfp_prediction.data_gathering.dataset import LFPDataset, MultiFeatureDataset

parser = argparse.ArgumentParser(description='Generic runner for NN models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['logging_params']['name'],
                              log_graph=True)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = nn_models[config['model_params']['name']](config['model_params'])

# experiment = Experiment(model, config['exp_params'])

data = LFPDataset(**config["data_params"], pin_memory=config['trainer_params']['gpus'] != 0)
# data = MultiFeatureDataset(**config["data_params"], pin_memory=config['trainer_params']['gpus'] != 0)


data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(log_momentum=True),
                     ModelCheckpoint(save_top_k=2,
                                     dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                     monitor="val_loss",
                                     save_last=True),
                 ],
                 strategy='ddp',
                 **config['trainer_params']
                 )

Path("{}/Samples".format(tb_logger.log_dir)).mkdir(exist_ok=True, parents=True)
Path("{}/Reconstructions".format(tb_logger.log_dir)).mkdir(exist_ok=True, parents=True)

print("======= Training {} =======".format(config['model_params']['name']))
runner.fit(model=model, datamodule=data)
