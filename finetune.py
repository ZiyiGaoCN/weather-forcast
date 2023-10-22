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

from dataset.dataset_npy import WeatherDataet_differentdata
from torch.utils.data import DataLoader

from dataset.transform.transform import Normalize, InverseNormalize
import datetime
import os

from rollout import Construct_Dataset
import tqdm

#Use a fixed seed for repreducible results
np.random.seed(1)
 
def set_seed(seed_value = 0):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)

@hydra.main(version_base=None, config_path='./cfgs/finetune', config_name="finetune2")
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

    logger = None 
    if hasattr(cfg,'logger'):
        config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        run = wandb.init(
            project=cfg.logger.project,
            name=cfg.logger.name,
            config=config)
        logger = wandb
    base_path = os.path.join(cfg.data.data_path,'dataset.npy')
    data_ori = np.memmap(base_path, dtype = 'float32',mode = 'c', shape = (14612, 70, 161, 161) , order = 'C')
    data_0 = np.memmap(base_path, dtype = 'float32',mode = 'c', shape = (14612, 70, 161, 161) , order = 'C')
    data_1 = np.memmap(base_path, dtype = 'float32',mode = 'c', shape = (14612, 70, 161, 161) , order = 'C')
    
    dataset = WeatherDataet_differentdata(
        target_data=data_ori,input_data1=data_0,input_data2=data_1,
        preload=False,range=(0,13148))
    valid = WeatherDataet_differentdata(
        target_data=data_ori,input_data1=data_0,input_data2=data_1,
        preload=False,range=(13148,14612))

    for i in range(1,20):
        checkpoint = torch.load(os.path.join(cfg.finetune.checkpoint_dir,f'save_{i}.pt'),map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        output_path = os.path.join(cfg.finetune.generated_dir,f'dataset_{i}.npy')
        raw_data = np.memmap(output_path, dtype = 'float32',mode = 'w+', shape = (14612, 70, 161, 161) , order = 'C')
        Construct_Dataset(model, dataloader=DataLoader(dataset, batch_size=64,shuffle=False, num_workers= 8, prefetch_factor=8),
                            data=raw_data, device=cfg.train.device,start = i + 1)
        Construct_Dataset(model, dataloader=DataLoader(valid, batch_size=64,shuffle=False, num_workers= 8, prefetch_factor=8),
                            data=raw_data, device=cfg.train.device,start = 13148 + i + 1)
        
        model = model.to('cpu')

        data_0 , data_1 = data_1, data_0
        data_1 = np.memmap(os.path.join(cfg.finetune.generated_dir,f'dataset_{i}.npy'), dtype = 'float32',mode = 'c', shape = (14612, 70, 161, 161) , order = 'C')
        dataset = WeatherDataet_differentdata(
            target_data=data_ori,input_data1=data_0,input_data2=data_1,
            preload=False,range=(i,13148))
        
        valid = WeatherDataet_differentdata(
            target_data=data_ori,input_data1=data_0,input_data2=data_1,
            preload=False,range=(13148+i,14612))

        losses = []
        for j in tqdm.tqdm(range(13148+i+2,14612)):
            input = np.array(raw_data[j,65:70,...])
            target = np.array(data_ori[j,65:70,...])
            loss = np.mean((input - target)**2,axis=(1,2))
            losses.append(loss)
        print(np.mean(losses,axis=0))
        loss = np.mean(losses,axis=0)
        
        table = wandb.Table(data=[loss], columns = [f'loss_{i}' for i in range(5)])
        logger.log({"val_losses": table})
        


        train(cfg.train, model, DataLoader(dataset, batch_size=cfg.train.dataloder.batch_size,shuffle=True, num_workers= 16, prefetch_factor=16)
              ,None,None, wandb=logger, inverse_transform_target = None)
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
        },os.path.join(cfg.finetune.checkpoint_dir,f'save_{i+1}.pt'))
        model = model.to('cpu')
        if i > 2:
            os.remove(os.path.join(cfg.finetune.generated_dir,f'dataset_{i-2}.npy'))

if __name__ == '__main__':
    main()