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

from train_deepspeed import train
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
import copy
import deepspeed
from deepspeed import comm

#Use a fixed seed for repreducible results
np.random.seed(1)
 
def set_seed(seed_value = 0):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)

# @hydra.main(version_base=None, config_path='./cfgs/finetune', config_name="finetune2_deepspeed")
def main(cfg:DictConfig):

    set_seed(cfg.seed)

    # using now() to get current time
    current_time = datetime.datetime.now()
    # with open_dict(cfg):
    #     cfg.train.save_file = f'./{cfg.train.ckpt_dir}/{current_time.strftime("%Y-%m-%d-%H")}'
    # if not os.path.exists(cfg.train.save_file):
    #     os.mkdir(cfg.train.save_file)
    
    print(cfg)
    
    # Initialize model 
    model_path = cfg.model.model_path
    model_type = cfg.model.model_type
    model_module = importlib.import_module(model_path)
    model_class = getattr(model_module, model_type)
    model = model_class(**cfg.model.param)

    logger = None 
    if hasattr(cfg,'logger') and cfg.deepspeed.local_rank == 0:
        config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        run = wandb.init(
            project=cfg.logger.project,
            name=cfg.logger.name,
            config=config)
        logger = wandb
    # if cfg.deepspeed.local_rank == 0:
    base_path = os.path.join(cfg.data.data_path,'dataset.npy')
    data_ori = np.memmap(base_path, dtype = 'float32',mode = 'c', shape = (14612, 70, 161, 161) , order = 'C')
    data_0 = np.memmap(base_path, dtype = 'float32',mode = 'c', shape = (14612, 70, 161, 161) , order = 'C')
    data_1 = np.memmap(base_path, dtype = 'float32',mode = 'c', shape = (14612, 70, 161, 161) , order = 'C')
    
    start_range = 0
        
    dataset = WeatherDataet_differentdata(
        target_data=data_ori,input_data1=data_0,input_data2=data_1,
        preload=False,range=(start_range,13148))
    valid = WeatherDataet_differentdata(
        target_data=data_ori,input_data1=data_0,input_data2=data_1,
        preload=False,range=(13148,14612))

    model_engine = None
    
    acc_loss =[]

    for i in range(0,20):
        if i > 0:
            comm.barrier() 
        checkpoint = torch.load(os.path.join(cfg.finetune.checkpoint_dir,f'save_{i}.pt'),map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if i==0:
            model_engine, optimizer, trainloader, _ = deepspeed.initialize(config=cfg.train.deepspeed_config,
                                                     model=model,
                                                     model_parameters=model.parameters(),)


        output_path = os.path.join(cfg.finetune.generated_dir,f'dataset_{i}.npy')
        if cfg.deepspeed.local_rank == 0 and i >0 :
            dataset = WeatherDataet_differentdata(
                target_data=data_ori,input_data1=data_0,input_data2=data_1,
                preload=False,range=(i,13148))
            raw_data = np.memmap(output_path, dtype = 'float32',mode = 'w+', shape = (14612, 70, 161, 161) , order = 'C')
            Construct_Dataset(model, dataloader=DataLoader(dataset, batch_size=64,shuffle=False, num_workers= 8, prefetch_factor=8),
                                data=raw_data, device=cfg.train.device,start = start_range + i + 1)
            Construct_Dataset(model, dataloader=DataLoader(valid, batch_size=64,shuffle=False, num_workers= 8, prefetch_factor=8),
                                data=raw_data, device=cfg.train.device,start = 13148 + i + 1)
        comm.barrier() 
        
        # if i < 4:
        checkpoint = torch.load(os.path.join(cfg.finetune.checkpoint_dir,f'save_{i}.pt'),map_location=torch.device('cpu'))
        # else :
        #     checkpoint = torch.load(os.path.join(cfg.finetune.checkpoint_dir,f'save_{i-1}.pt'),map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to('cpu')

        data_0 , data_1 = data_1, data_0
        if i > 0:
            data_1 = np.memmap(os.path.join(cfg.finetune.generated_dir,f'dataset_{i}.npy'), dtype = 'float32',mode = 'c', shape = (14612, 70, 161, 161) , order = 'C')
        else:
            data_1 = np.memmap(base_path, dtype = 'float32',mode = 'c', shape = (14612, 70, 161, 161) , order = 'C')
        
        dataset = WeatherDataet_differentdata(
                target_data=data_ori,input_data1=data_0,input_data2=data_1,
                preload=False,range=(start_range+i,13148))
        

        # if i % 2 == 1:
        #     dataset = WeatherDataet_differentdata(
        #         target_data=data_ori,input_data1=data_0,input_data2=data_1,
        #         preload=False,range=(i,6574))
        # else:
        #     dataset = WeatherDataet_differentdata(
        #         target_data=data_ori,input_data1=data_0,input_data2=data_1,
        #         preload=False,range=(6574+i,13148))
        
        valid = WeatherDataet_differentdata(
            target_data=data_ori,input_data1=data_0,input_data2=data_1,
            preload=False,range=(13148+i,14612))

        if cfg.deepspeed.local_rank == 0 and i > 0 :
            losses = []
            for j in tqdm.tqdm(range(13148+i+2,14612)):
                input = np.array(raw_data[j,65:70,...])
                target = np.array(data_ori[j,65:70,...])
                loss = np.mean((input - target)**2,axis=(1,2))
                losses.append(loss)
            print(np.mean(losses,axis=0))
            loss = np.mean(losses,axis=0)

            acc_loss.append(np.concatenate((np.array([i]),loss),axis=0))

            table = wandb.Table(data=acc_loss, columns = ['step'] + [f'loss_{i}' for i in range(5)])
            logger.log({"val_losses": table})
        
        # model2 = model_class(**cfg.model.param)
        # model2.load_state_dict(copy.deepcopy(model.state_dict()))

        model_engine = train(cfg.train, model,  dataset
              ,None,None, wandb=logger, inverse_transform_target = None,
              deepspeed_config=cfg.deepspeed,offset=i+1)
        
        client_sd = {
            "step":None
        }
        ckpt_id = f'hello'
        path =os.path.join(cfg.finetune.checkpoint_dir,'latest')
        model_engine.save_checkpoint(path, ckpt_id)  
        

        if cfg.deepspeed.local_rank == 0:
            checkpoint = torch.load(os.path.join(cfg.finetune.checkpoint_dir,'latest/hello/mp_rank_00_model_states.pt'),map_location='cpu')
            
            torch.save({
                "model_state_dict": checkpoint['module'],
                "optimizer_state_dict": None,
            },os.path.join(cfg.finetune.checkpoint_dir,f'save_{i+1}.pt'))
            model = model.to('cpu')
            if i > 2:
                os.remove(os.path.join(cfg.finetune.generated_dir,f'dataset_{i-2}.npy'))

import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hello")
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    
    cfg_path = './cfgs/finetune/finetune2_deepspeed.yaml'

    with open(cfg_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Convert to omegaconf.DictConfig
    config = OmegaConf.create(yaml_data)
    config_deepspeed = OmegaConf.create({"deepspeed":vars(args)})
    config = OmegaConf.merge(config, config_deepspeed)
    main(config)