"""
This script demonstrates initialisation, training, evaluation, and forecasting of ForecastNet. The dataset used for the
time-invariance test in section 6.1 of the ForecastNet paper is used for this demonstration.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf , open_dict
import importlib

from train import train
from evaluate import evaluate
from dataHelpers import generate_data
from dataset.loading import loading
import wandb

from dataset.dataset_mem import split_dataset
from torch.utils.data import DataLoader

from dataset.transform.transform import Normalize, InverseNormalize
import datetime
import os

#Use a fixed seed for repreducible results
np.random.seed(1)
 
def set_seed(seed_value = 0):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)

@hydra.main(version_base=None, config_path='./cfgs', config_name="vanilla_transformer")
def main(cfg:DictConfig):

    set_seed(cfg.seed)

    # using now() to get current time
    current_time = datetime.datetime.now()
    with open_dict(cfg):
        cfg.train.save_file = f'./{cfg.train.ckpt_dir}/{current_time.strftime("%Y-%m-%d-%H-%M-%S")}'
    if not os.path.exists(cfg.train.save_file):
        os.mkdir(cfg.train.save_file)
    
    print(cfg)
    
    # Initialize model 
    model_path = cfg.model.model_path
    model_type = cfg.model.model_type
    model_module = importlib.import_module(model_path)
    model_class = getattr(model_module, model_type)
    model = model_class(**cfg.model.param)

    if cfg.logger is not None:
        wandb.init(project=cfg.logger.project, name=cfg.logger.name)
        # wandb.config.update(cfg)

    # fcstnet = forecastNet(in_seq_length=in_seq_length, out_seq_length=out_seq_length, input_dim=input_dim,
    #                         hidden_dim=hidden_dim, output_dim=output_dim, model_type = model_type, batch_size = batch_size,
    #                         n_epochs = n_epochs, learning_rate = learning_rate, weight_decay=1e-5, 
    #                         save_file = './forecastnet.pt',device = "cuda:1")

    # transform_train = Normalize('../dataset/normalization', normalization_type = 'TAO', is_target = False)
    # transform_target = Normalize('../dataset/normalization',  normalization_type = 'TAO', is_target = True)
    # inverse_transform_target = InverseNormalize('../dataset/normalization',  normalization_type = 'TAO', is_target = True)

    train_data, valid_data = split_dataset(cfg.data.data_path,ratio=0.8,
                                        transform=None,
                                        target_transform=None)
    train_dataloader = DataLoader(train_data, batch_size=cfg.data.batch_size_train, shuffle=True, num_workers=16)
    valid_dataloader = DataLoader(valid_data, batch_size=cfg.data.batch_size_val, shuffle=False, num_workers=16)

    # Train the model
    training_costs, validation_costs = train(cfg.train,model, train_dataloader,valid_dataloader, wandb=wandb, inverse_transform_target = None)
    

if __name__ == '__main__':
    main()