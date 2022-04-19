import argparse
import os
from pathlib import Path

import yaml
from pytorch_lightning.strategies import DDPStrategy

from lfp_prediction.models import nn_models
from experiment import Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lfp_prediction.data_gathering.dataset import LFPDataset


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
                              name=config['model_params']['name'],
                              log_graph=True)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = nn_models[config['model_params']['name']](**config['model_params'])

experiment = Experiment(model, config['exp_params'])

data = LFPDataset(**config["data_params"], pin_memory=config['trainer_params']['gpus'] != 0)

data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2,
                                     dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                     monitor="val_loss",
                                     save_last=True),
                 ],
                 strategy=DDPStrategy(find_unused_parameters=False),
                 **config['trainer_params']
                 )

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
