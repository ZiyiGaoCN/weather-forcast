"""
This script demonstrates initialisation, training, evaluation, and forecasting of ForecastNet. The dataset used for the
time-invariance test in section 6.1 of the ForecastNet paper is used for this demonstration.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

from http import client

from pandas.core.indexes import multi
import weather_forcast
from weather_forcast.utils import initial_model
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf , open_dict
import importlib

import torch.nn.functional as F

from train_deepspeed import train
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

from dataset.dataset_fengwu import Replay_buffer_fengwu
from loguru import logger as loguru_logger
from dataset.dataset_torch import WeatherDataet_numpy
from validate import validate_20step
import loguru

from weather_forcast.utils import get_stds, log_indices, get_loss_weights

#Use a fixed seed for repreducible results
np.random.seed(1)
 
def set_seed(seed_value = 0):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)

def main(cfg:DictConfig):

    set_seed(cfg.seed)
    loss_weights_gc = get_loss_weights()
    
    # Initialize model 
    # model = initial_model(cfg.model)

    if cfg.finetune.initialize_checkpoint is not None:
        
        model_weight = torch.load(cfg.finetune.initialize_checkpoint)
        if model_weight.get('model_params',None):
            model = initial_model(model_weight['model_params'])
            loguru.logger.info("loading from checkpoint")
            loguru.logger.info(model_weight['model_params'])
            client_sd = {'model_params': model_weight['model_params']}
        else:
            loguru.logger.info("loading from config")
            loguru.logger.info(cfg.model)
            model = initial_model(cfg.model)
            client_sd = {'model_params': cfg.model}
        model.load_state_dict(model_weight['module'])
        del model_weight
    else: 
        raise NotImplementedError

    if cfg.train.uncertainty_loss == False:
        model.uncertainty_loss = False

    with open_dict(cfg):
        cfg.train.save_file = f'{cfg.train.ckpt_dir}/{cfg.logger.name}'
    try:
        os.mkdir(cfg.train.save_file)
    except:
        loguru.logger.warning('dir exists')

    if hasattr(cfg,'logger') and cfg.deepspeed.local_rank == 0:
        config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        run = wandb.init(
            project=cfg.logger.project,
            name=cfg.logger.name,
            config=config)
        logger = wandb
    else:
        logger = None 

    base_path = cfg.data.npy_name
    data_ori = np.memmap(base_path, dtype = 'float32',mode = 'c', shape = tuple(cfg.data.shape) , order = 'C')
    
    large_buffer_folder = os.path.join(cfg.data.large_buffer_folder,cfg.logger.name)
    small_buffer_folder = os.path.join(cfg.data.buffer_folder,cfg.logger.name)
    try: 
        os.mkdir(large_buffer_folder)
    except:
        loguru.logger.warning('large_buffer_folder dir exists')
    
    try: 
        os.mkdir(small_buffer_folder)
    except:
        loguru.logger.warning('small_buffer_folder dir exists')
    
    
    large_buffer_file = os.path.join(large_buffer_folder,'replaybuffer.npy')
    # small_buffer_folder = os.path.join()
    
    loguru.logger.info("Creating replay buffer")
    
    Replay_Buffer = Replay_buffer_fengwu(
        data=data_ori, train_range=cfg.data.train_range, buffer_file = large_buffer_file,
        small_buffer_folder = small_buffer_folder,
        buffer_size=cfg.data.buffer_size, shape = cfg.data.shape,
        uncertainty_finetune = cfg.train.uncertainty_finetune,
    )
    loguru.logger.info("Replay buffer created")
    
    train_valid_set = WeatherDataet_numpy(data_ori,range=[55000,55480],step=20)
    train_validloader = DataLoader(train_valid_set, batch_size=32, shuffle=False, num_workers=8)

    valid_set = WeatherDataet_numpy(data_ori,range=cfg.data.val_range,step=20)
    validloader = DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=8)

    step = 0
    
    deepspeed_config_normal = cfg.train.deepspeed_config_yaml
    deepspeed_config_dict = OmegaConf.to_container(deepspeed_config_normal, resolve=True)
    
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(config=deepspeed_config_dict,
                                                    model=model,
                                                    model_parameters=model.parameters(),
                                                    training_data=None)
    
    train_param = cfg.train
    
    if cfg.train.optimizer.scheduler.name == 'NoamLR':
        from scheduler.NOAM import NOAMLR
        scheduler = NOAMLR(optimizer, warmup_steps=cfg.train.optimizer.scheduler.warmup_steps, model_size=cfg.train.optimizer.scheduler.hidden_dim)
    elif cfg.train.optimizer.scheduler.name == 'CosineAnnealingLR':
        from weather_forcast.scheduler.Cosine import CosineLR
        scheduler = CosineLR(optimizer, num_epochs=train_param.optimizer.scheduler.T_max, 
                             eta_min=train_param.optimizer.scheduler.eta_min,
                             warmup_steps=train_param.optimizer.scheduler.warmup_steps)    
    else:
        scheduler = None

    train_param = cfg.train
    deepspeed_config = cfg.deepspeed

    

    for epoch in range(cfg.train.n_epochs):
        if epoch > 0:
            comm.barrier() 
        
        loguru.logger.info(f'Epoch: {epoch + 1} of {cfg.train.n_epochs}')
        loguru.logger.info('Sampling data')

        if hasattr(cfg.data,'sample'):
            if cfg.data.sample.name == 'Poisson':
                from weather_forcast.sample.poisson import distribution
                t = epoch/10
                wandb.log({
                    "sample_ratio": t,
                })
                sample_ratio = distribution(t)
            elif cfg.data.sample.name == 'yl':
                from weather_forcast.sample.yl import distribution,get_param
                t = cfg.data.sample.n*(epoch/cfg.train.n_epochs)
                p = get_param(t)[0]
                wandb.log({
                    "sample_ratio": t,
                })
                sample_ratio = distribution(p)
            elif cfg.data.sample.name == 'fengwu':
                from weather_forcast.sample.fengwu_sample import distribution
                sample_ratio = distribution()
            else:
                sample_ratio = None
        else :
            sample_ratio= None 
        loguru.logger.info(f'sample_ratio:{sample_ratio}')

        dataset = Replay_Buffer.build_dataset(size=cfg.data.epoch_size,sample_ratio=sample_ratio,
                                              multi_step=cfg.train.multi_step)
        loguru.logger.info('Data sampled')
                
        trainloader = model_engine.deepspeed_io(dataset)
        model_engine.dataloader = trainloader

        r_step = 0
        model.train()
        for i,(input,target,time, time_step, time_embed) in enumerate(tqdm.tqdm(trainloader)):
            # Send input and output data to the GPU/CPU
            input = input.to(model_engine.device)
            target = target.to(model_engine.device)
            # Compute outputs and loss for a mixture density network output
            
            # loguru.logger.info("Input generated")
            
            B, in_seq, C_in, H, W = input.shape
            B, out_seq, C_out, H, W = target.shape
            
            # time_embed = 1 if model.time_embed else None
            # if cfg.train.multi_step: 
            #     time_embed = torch.randint(1, 5, (B,))
            # else:
            #     time_embed = 1 if model.time_embed else None
            
            input = input.view(B, in_seq*C_in, H, W)
            target = target.view(B, out_seq*C_out, H, W)
            
            if train_param.uncertainty_loss:
                tu = model_engine(input,time_embed)
                outputs, sigma = tu[0], tu[1]
                
            else:
                outputs = model_engine(input,time_embed)
                sigma = None
            loss = F.mse_loss(input=outputs, target=target, reduction='none')
            uplow_loss = None
            
            if train_param.time_regularization: 
                raise NotImplementedError
                with torch.no_grad():
                    time_embed_half = 0.5 
                    half_output = model_engine(input,time_embed_half)
                outputs_regularization = model_engine(half_output,time_embed_half)
                target_output = outputs.detach()
                loss_regularization = F.mse_loss(input=outputs_regularization, target=target_output)
            
            if train_param.time_regularization:
                raise NotImplementedError
                compute_loss = loss + train_param.time_regularization_weight * loss_regularization
            else:
                if sigma is None:
                    if hasattr(train_param,'loss_weights_gc') and train_param.loss_weights_gc:
                        compute_loss = (loss * torch.from_numpy(loss_weights_gc).to(loss.device).reshape(1,-1,1,1)).sum(dim=1)
                        
                    else:
                        compute_loss = loss
                else:
                    if train_param.meaning_uncertainty:
                        use_sigma = sigma.mean(dim=(2,3),keepdim=True)
                    else:
                        use_sigma = sigma
                        
                    if hasattr(train_param,'em_uncertainty_training') and train_param.em_uncertainty_training.enable:   
                        if (epoch + 1) % train_param.em_uncertainty_training.uncertainty_graident_everyepoch == 0:
                            loss = loss.detach()
                            compute_loss = loss / (2*torch.exp(2*use_sigma)) + use_sigma
                        else:
                            use_sigma = use_sigma.detach()
                            compute_loss = loss / (2*torch.exp(2*use_sigma)) + use_sigma
                    else:            
                        compute_loss = loss / (2*torch.exp(2*use_sigma)) + use_sigma
                        uplow_loss = - 0.01 * tu[2] + 0.01 * tu[3] 
                        uplow_loss = uplow_loss.mean() 
            
            compute_loss = compute_loss.mean()
            if uplow_loss is not None:
                compute_loss += uplow_loss

            model_engine.backward(compute_loss)
            
            # if step % 100 == 0:
            #     comm.barrier()
            #     grads = [torch.sum(p.grad**2).item() for p in model.parameters() if (p is not None) and p.requires_grad]
            #     grad_norm = np.sqrt(sum(grads))
            #     if wandb is not None and deepspeed_config.local_rank == 0:
            #         wandb.log({
            #             'grad_norm': grad_norm,
            #         }, commit=False)
                    
            model_engine.step() 
            
            # time += 1
            if not model.time_embed:
                time += 1
                time_step += 1
            else :
                time += time_embed
                time_step = time_step + time_embed
            Replay_Buffer.add_buffer_20step(
                outputs.detach().cpu().numpy(), time.detach().cpu().numpy(), time_step.detach().cpu().numpy(),
                sigma=np.exp(sigma_mean.detach().cpu().numpy()) if hasattr(cfg.train,'uncertainty_finetune') and cfg.train.uncertainty_finetune else None
                )
            
            r_step += 1
            if r_step % model_engine.gradient_accumulation_steps() == 0:
                r_step = 0
                step +=1
                if scheduler is not None: 
                    scheduler.step()   
            
                if wandb is not None and deepspeed_config.local_rank == 0 :
                    wandb.log({
                        'train_loss': loss.detach().mean().item(),
                        'lr': optimizer.param_groups[0]['lr']
                    })
                    if model.uncertainty_loss:
                        wandb.log({
                            # 'train_loss_sigma': torch.exp(sigma.item()),
                            'computed_loss': compute_loss.item()
                        },commit=False)
                        sigma_mean = torch.exp(sigma).mean(axis=(0,2,3))
                        upload = {
                                'uncertainty/2m': sigma_mean[-1],
                                'uncertainty/10v': sigma_mean[-2],
                                'uncertainty/10u': sigma_mean[-3],
                                'uncertainty/z500': sigma_mean[45],
                                'uncertainty/t850': sigma_mean[10],
                            }
                        wandb.log(upload, commit=False)
                    if train_param.time_regularization:
                        raise NotImplementedError 
                        wandb.log({
                            'train_loss_time_embed': loss_regularization.item(),
                        },commit=False)
                if hasattr(train_param,'validate_step') and step % train_param.validate_step == 0 and hasattr(train_param,'save_file'):
                
                    client_sd['step'] = step
                    ckpt_id = f'step_{step}'
                    model_engine.save_checkpoint(train_param.save_file, ckpt_id,client_sd)

            
                if step % cfg.train.validate_step == 0 and cfg.deepspeed.local_rank == 0:
                    if valid_set is not None:
                        train_param_20, train_acc_20 = validate_20step(model,train_validloader)
                        params_20,acc_20 = validate_20step(model,validloader)
                        model.train()
                        if wandb is not None:
                            
                            day = {
                                "6h" : 0, 
                                "day1": 3, 
                                "day3": 11,
                                "day5": 19,
                            }
                            
                            for k,v in day.items():
                                params = params_20[v]
                                train_params = train_param_20[v]

                                upload = {
                                    f'{k}/2m': params[-1],
                                    f'{k}/10v': params[-2],
                                    f'{k}/10u': params[-3],
                                    f'{k}/z500': params[45],
                                    f'{k}/t850': params[10],
                                }
                                wandb.log(upload, commit=False)
                                upload_train = {
                                    f'{k}/train_2m': train_params[-1],
                                    f'{k}/train_10v': train_params[-2],
                                    f'{k}/train_10u': train_params[-3],
                                    f'{k}/train_z500': train_params[45],
                                    f'{k}/train_t850': train_params[10],
                                } 
                                wandb.log(upload_train, commit=False)
                                
                

import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hello")
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--cfg_path', default=None, type=str, help='base config')
    parser.add_argument('--cfg_new', default=None, type=str, help='new config')
    
    args = parser.parse_args()
    
    with open(args.cfg_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    loguru.logger.info(args.cfg_new)
    if args.cfg_new is not None:
        loguru.logger.info(f'load overiding config from {args.cfg_new}')
        with open(args.cfg_new) as file:
            new_yaml_data = yaml.safe_load(file)

    # Convert to omegaconf.DictConfig
    config = OmegaConf.create(yaml_data)
    # config_deepspeed = 
    config = OmegaConf.merge(config, OmegaConf.create({"deepspeed":vars(args)}))
    
    config = OmegaConf.merge(config, OmegaConf.create(new_yaml_data))
    
    loguru.logger.info(config)
    
    main(config)