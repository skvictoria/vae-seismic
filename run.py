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
# from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
#from dataset import VAEDataset
from f3_dataloader_lightning import VAEDataset
from pytorch_lightning.strategies import DDPStrategy
# from pytorch_lightning.plugins import DDPPlugin

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

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()

tb_logger =  TensorBoardLogger("./logs/",
                               name="ConditionalVAE",)

runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPStrategy(find_unused_parameters=False),
                 accelerator="gpu", max_epochs = config["trainer_params"]["max_epochs"] )


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Test_Input").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Test_Label").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/mean_image").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/std_image").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)



# !!!!! Check Later !!!!!
# For reproducibility
# seed_everything(1265, True)

# model = vae_models['ConditionalVAE'](  in_channels=1,
#   num_classes= 10,
#   latent_dim= 128,
#   img_size=28)
# experiment = VAEXperiment(model,
#                           params={
#     "LR": 0.005,
#     "weight_decay": 0.0,
# #   "scheduler_gamma": 0.95,
#     "kld_weight": 0.00025,
#     "manual_seed": 1265
# })

# data = VAEDataset(  data_path= 'E:/celeba/',
#   train_batch_size= 64,
#   val_batch_size=  64,
#   patch_size= 64,
#   num_workers= 4, pin_memory=True)
