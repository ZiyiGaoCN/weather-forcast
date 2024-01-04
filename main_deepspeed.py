"""
This script demonstrates initialisation, training, evaluation, and forecasting of ForecastNet. The dataset used for the
time-invariance test in section 6.1 of the ForecastNet paper is used for this demonstration.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import datetime
import os
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf , open_dict
import yaml
import importlib

from train_deepspeed import train
from dataset.loading import loading
import wandb

from dataset.dataset_torch import WeatherDataet_numpy
from dataset.transform.transform import Normalize, InverseNormalize
from utils import initial_model
import loguru 

#Use a fixed seed for repreducible results
np.random.seed(1)
 
def set_seed(seed_value = 0):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)

# @hydra.main(version_base=None, config_path='./cfgs', config_name="swin_AR_deepspeed")
def main(cfg:DictConfig):

    set_seed(cfg.seed)

    # using now() to get current time
    current_time = datetime.datetime.now()
    with open_dict(cfg):
        cfg.train.save_file = f'{cfg.train.ckpt_dir}/{cfg.logger.name}'
    try:
        os.mkdir(cfg.train.save_file)
    except:
        loguru.logger.warning('dir exists')
        pass
    print(cfg)
    
    model = initial_model(cfg.model)

    loguru.logger.info(model)

    if hasattr(cfg,'logger') and cfg.deepspeed.local_rank == 0:
        config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        run = wandb.init(
            project=cfg.logger.project,
            name=cfg.logger.name,
            config=config)
        
        # wandb.config.update(cfg)

    data = np.memmap(cfg.data.npy_name, dtype=np.float32, mode='c', shape = tuple(cfg.data.shape))

    train_set = WeatherDataet_numpy(data,range=cfg.data.train_range,input_step=cfg.data.input_step)
    valid_set = WeatherDataet_numpy(data,range=cfg.data.val_range,input_step=cfg.data.input_step)
    

    # train_data, valid_data, valid_data_20step = split_dataset_npy(**cfg.data, autoregressive=cfg.train.autoregressive,transform=None, target_transform=None)
    engine = train(cfg.train,cfg.model,model, train_set,valid_set, wandb=wandb, 
                                             inverse_transform_target = None,
                                             deepspeed_config=cfg.deepspeed,finetune_time=1)
    
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="hello")
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--cfg_path', default='./cfgs/swin_AR_deepspeed.yaml', type=str, help='node rank for distributed training')
    args = parser.parse_args()
    
    # cfg_path = './cfgs/swin_AR_deepspeed.yaml'

    with open(args.cfg_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        config = OmegaConf.create(yaml_data)
        config_deepspeed = OmegaConf.create({"deepspeed":vars(args)})
        config = OmegaConf.merge(config, config_deepspeed)
        
    main(config)