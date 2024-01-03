"""
A training function for ForecastNet. This code could be improved by using a PyTorch dataloader

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import numpy as np
from sympy import Not
import torch
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
import tqdm
import os
import pandas as pd
import deepspeed
from deepspeed import comm
from omegaconf import DictConfig, OmegaConf , open_dict 

from weather_forcast.utils import get_stds, log_indices

from weather_forcast.validate import validate_onestep
import loguru 

def train(train_param,model_param, model, train_set, valid_set=None,valid_set_20step=None,wandb=None, inverse_transform_target=None,
          deepspeed_config=None, finetune_time = 1):
    indices = log_indices()
    stds = get_stds()

    loguru.logger.info(f'deepspeed_config: {deepspeed_config}')

    deepspeed_config_normal = train_param.deepspeed_config_yaml
    deepspeed_config_dict = OmegaConf.to_container(deepspeed_config_normal, resolve=True)
    

    model_engine, optimizer, trainloader, _ = deepspeed.initialize(config=deepspeed_config_dict,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     training_data=train_set)
    if train_param.optimizer.scheduler.name == 'NoamLR':
        from scheduler.NOAM import NOAMLR
        scheduler = NOAMLR(optimizer, warmup_steps=train_param.optimizer.scheduler.warmup_steps, scale=1/finetune_time, model_size=train_param.optimizer.scheduler.hidden_dim)
    elif train_param.optimizer.scheduler.name == 'CosineAnnealingLR':
        from weather_forcast.scheduler.Cosine import CosineLR
        scheduler = CosineLR(optimizer, num_epochs=train_param.optimizer.scheduler.T_max, 
                             eta_min=train_param.optimizer.scheduler.eta_min,
                             warmup_steps=train_param.optimizer.scheduler.warmup_steps)    
    else:
        scheduler = None

    if valid_set is not None:
        validloader = DataLoader(valid_set, **train_param.validate_dataloader)
    
    if hasattr(train_param,'restore_ression') and train_param.restore_session.yes is True:
        # Load model parameters
        _, client_sd = model_engine.load_checkpoint(train_param.restore_session.load_dir, train_param.restore_session.ckpt_id)
        step = client_sd['step']
    else :
        step = 0
        client_sd = {
            'step': 0
        }

    client_sd['model_params'] = model_param

    r_step = 0
    
    for epoch in range(train_param.n_epochs):
        trainloader.data_sampler.set_epoch(epoch+finetune_time*100)
        # Start the epoch timer
        t_start = time.time()

        # Print the epoch number
        print('Epoch: %i of %i' % (epoch + 1, train_param.n_epochs))

        batch_cost = []

        count = 0

        model.train()
        for i,(input,target) in enumerate(tqdm.tqdm(trainloader)):
            # Send input and output data to the GPU/CPU
            input = input.to(model_engine.device)
            target = target.to(model_engine.device)

            # Compute outputs and loss for a mixture density network output
            
            B, in_seq, C_in, H, W = input.shape
            B, out_seq, C_out, H, W = target.shape
            
            time_embed = 1 if model.time_embed else None
            
            input = input.view(B, in_seq*C_in, H, W)
            target = target.view(B, out_seq*C_out, H, W)
            
            if train_param.uncertainty_loss:
                outputs, sigma = model_engine(input,time_embed)
            else:
                outputs = model_engine(input,time_embed)
                sigma = None
            loss = F.mse_loss(input=outputs, target=target, reduction='none')

            
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
                    compute_loss = loss
                else:
                    if train_param.meaning_uncertainty:
                        use_sigma = sigma.mean(dim=(2,3),keepdim=True)
                    else:
                        use_sigma = sigma
                        
                    if train_param.em_uncertainty_training:
                        if train_param.em_uncertainty_training.enable:
                            if (epoch + 1) % train_param.em_uncertainty_training.uncertainty_graident_everyepoch == 0:
                                loss = loss.detach()
                                compute_loss = loss / (2*torch.exp(2*use_sigma)) + use_sigma
                            else:
                                use_sigma = use_sigma.detach()
                                compute_loss = loss / (2*torch.exp(2*use_sigma)) + use_sigma
                    else:            
                        compute_loss = loss / (2*torch.exp(2*use_sigma)) + use_sigma 
            
            compute_loss = compute_loss.mean()

            batch_cost.append(compute_loss.item())
            model_engine.backward(compute_loss)
            
            
            
            if step % 100 == 1 and r_step% model_engine.gradient_accumulation_steps() == 0:
                comm.barrier()
                grads = [torch.sum(p.grad**2).item() for p in model.parameters() if p.requires_grad]
                grad_norm = np.sqrt(sum(grads))
                if wandb is not None and deepspeed_config.local_rank == 0:
                    wandb.log({
                        'grad_norm': grad_norm,
                    }, commit=False)
            
            
            r_step += 1
            model_engine.step() 
            if r_step % model_engine.gradient_accumulation_steps() == 0:
                r_step = 0
                step +=1
                scheduler.step()         
            
                
                if wandb is not None and deepspeed_config.local_rank == 0 :
                    loss = loss.detach().mean(dim=(0,2,3))
                    log_info = {
                        f'train_loss/{k}': torch.sqrt(loss[v]).item() * stds [v]
                        for k,v in indices.items()
                    }
                    log_info ['train_loss']=loss.mean().item()
                    log_info ['lr'] = optimizer.param_groups[0]['lr']
                    
                    wandb.log(log_info)
                    
                    if train_param.uncertainty_loss:
                        assert model.uncertainty_loss == True
                        sigma_mean = sigma.mean(dim=(2,3),keepdim=True)
                        wandb.log({
                            'computed_loss': compute_loss.item()
                        },commit=False)
                        sigma_mean_batch = torch.exp(sigma_mean).mean(axis=(0))
                        # upload = {
                        #         'uncertainty/2m': sigma_mean[-1],
                        #         'uncertainty/10v': sigma_mean[-2],
                        #         'uncertainty/10u': sigma_mean[-3],
                        #         'uncertainty/z500': sigma_mean[45],
                        #         'uncertainty/t850': sigma_mean[10],
                        #     }
                        upload = {
                            f'uncertainty/{k}': sigma_mean_batch[v].item() * stds[v]
                            for k,v in indices.items()
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

                if step % train_param.validate_step == 0 and deepspeed_config.local_rank == 0:
                    if valid_set is not None:
                        rmse, acc = validate_onestep(model,validloader)
                        if wandb is not None:
                            
                            # upload = {
                            #     'valid_loss/2m': rmse[-1],
                            #     'valid_loss/10v': rmse[-2],
                            #     'valid_loss/10u': rmse[-3],
                            #     'valid_loss/z500': rmse[45],
                            #     'valid_loss/t850': rmse[10],
                            # }
                            upload = {
                                f'valid_loss/{k}': rmse[v].item()
                                for k,v in indices.items()
                            }
                            wandb.log(upload, commit=False)
                        model.train()

                        
                        

        epoch_cost = np.mean(batch_cost)

        print("Average epoch training cost: ", epoch_cost)
        print("Epoch time:                   %f seconds" % (time.time() - t_start))
        print("Estimated time to complete:   %.2f minutes, (%.2f seconds)" %
              ((train_param.n_epochs - epoch - 1) * (time.time() - t_start) / 60,
               (train_param.n_epochs - epoch - 1) * (time.time() - t_start)))

          

    return model_engine
