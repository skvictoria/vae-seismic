import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from f3_dataloader_lightning import VAEDataset
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl

from models import *  # Assuming this includes your model classes

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/cvae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])


trainer = pl.Trainer()
trainer.test(datamodule=data)

# Path to your `.ckpt` file
checkpoint_path = 'logs/ConditionalVAE/version_74/checkpoints/last.ckpt'

# Load the model
#model = vae_models[config['model_params']['name']](**config['model_params'])
#experiment = VAEXperiment(model,config['exp_params']).load_from_checkpoint(checkpoint_path)

experiment = VAEXperiment.load_from_checkpoint(checkpoint_path)

# Example: Using the model for inference
experiment.eval()
experiment.sample_one_image()
# # Assuming a dummy input tensor
# input_tensor = torch.randn((64, 1, 64, 64))  # Adjust the size based on your model's expected input
# with torch.no_grad():
#     output = model(input_tensor)
