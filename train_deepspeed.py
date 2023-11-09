"""
A training function for ForecastNet. This code could be improved by using a PyTorch dataloader

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import numpy as np
import torch
import time
import torch.nn.functional as F
import tqdm
import os
import pandas as pd
import deepspeed
from omegaconf import DictConfig, OmegaConf , open_dict 
    
from torch.utils.data import DataLoader
from validate import validate_onestep

def train(train_param, model, train_set, valid_set=None,valid_set_20step=None,wandb=None, inverse_transform_target=None,
          deepspeed_config=None, finetune_time = 1):
    # if train_param.optimizer.name == 'Adam':
    #     optimizer = torch.optim.Adam(model.parameters(), lr=train_param.optimizer.learning_rate, weight_decay=train_param.optimizer.weight_decay)
    # dict_param = OmegaConf.to_container(
    #     train_param, resolve=True, throw_on_missing=True
    # )

    model_engine, optimizer, trainloader, _ = deepspeed.initialize(config=train_param.deepspeed_config,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     training_data=train_set)
    if train_param.optimizer.scheduler.name == 'NoamLR':
        from scheduler.NOAM import NOAMLR
        scheduler = NOAMLR(optimizer, warmup_steps=train_param.optimizer.scheduler.warmup_steps, scale=1/finetune_time, model_size=train_param.optimizer.scheduler.hidden_dim)
    else :
        scheduler = None

    if valid_set is not None:
        validloader = DataLoader(valid_set, batch_size=train_param.batch_size, shuffle=False, num_workers=8, prefetch_factor=8)
    
    if hasattr(train_param,'restore_ression') and train_param.restore_session.yes is True:
        # Load model parameters
        _, client_sd = model_engine.load_checkpoint(train_param.restore_session.load_dir, train_param.restore_session.ckpt_id)
        step = client_sd['step']
    else :
        step = 0
        client_sd = {
            'step': 0
        }

    # Number of batch samples
    # n_samples = train_x.shape[0]

    # List to hold the training costs over each epoch
    training_costs = []
    validation_costs = []

    # Training loop
    for epoch in range(train_param.n_epochs):
        trainloader.data_sampler.set_epoch(epoch+finetune_time*100)
        # Start the epoch timer
        t_start = time.time()

        # Print the epoch number
        print('Epoch: %i of %i' % (epoch + 1, train_param.n_epochs))

        batch_cost = []

        count = 0

        # model.train()
        for i,(input,target) in enumerate(tqdm.tqdm(trainloader)):
            # Send input and output data to the GPU/CPU
            input = input.to(model_engine.device)
            target = target.to(model_engine.device)

            # Compute outputs and loss for a mixture density network output
            
            B, in_seq, C_in, H, W = input.shape
            B, out_seq, C_out, H, W = target.shape
            input = input.view(B, in_seq*C_in, H, W)
            target = target.view(B, out_seq*C_out, H, W)
            outputs = model_engine(input)
            loss = F.mse_loss(input=outputs, target=target)
            batch_cost.append(loss.item())
            model_engine.backward(loss)
            model_engine.step()
            
            step += 1
            scheduler.step()        
            if wandb is not None and deepspeed_config.local_rank == 0:
                wandb.log({'loss': loss.item()})
                wandb.log({'lr': optimizer.param_groups[0]['lr']})
            
            if hasattr(train_param,'validate_step') and step % train_param.validate_step == 0 and hasattr(train_param,'save_file'):
                
                client_sd['step'] = step
                ckpt_id = f'step_{step}'
                model_engine.save_checkpoint(train_param.save_file, ckpt_id)

            if step % train_param.validate_step == 0 and deepspeed_config.local_rank == 0:
                if valid_set is not None:
                    valid_loss, five_param = validate_onestep(model,validloader)
                    if wandb is not None:
                        wandb.log({'valid_loss': valid_loss})
                        print('valid_loss:',valid_loss)
                        print('five param:',five_param)
                        variable = ['t2m', 'u10', 'v10', 'msl', 'tp']
                        
                        for i in range(5):
                            wandb.log({f'valid_{variable[i]}': five_param[i]})
                    

        epoch_cost = np.mean(batch_cost)

        print("Average epoch training cost: ", epoch_cost)
        print("Epoch time:                   %f seconds" % (time.time() - t_start))
        print("Estimated time to complete:   %.2f minutes, (%.2f seconds)" %
              ((train_param.n_epochs - epoch - 1) * (time.time() - t_start) / 60,
               (train_param.n_epochs - epoch - 1) * (time.time() - t_start)))

          

    return model_engine
