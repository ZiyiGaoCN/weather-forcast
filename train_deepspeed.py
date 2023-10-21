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
    

# Set plot_train_progress to True if you want a plot of the forecast after each epoch
# plot_train_progress = False
# if plot_train_progress:
    # import matplotlib.pyplot as plt

def compute_rmse(output,target):
    '''
        result: (batch x lat x lon), eg: N x H x W
        target: (batch x lat x lon), eg: N x H x W
    '''
    
    B = 30 
    output = output[:,B:-B,B:-B]
    target = target[:,B:-B,B:-B]
    
    square = F.mse_loss(input=output, target=target, reduction='none')
    square = torch.mean(square,dim=(1,2))
    
    result = torch.sqrt(square)
    result = torch.mean(result)
    return result.item()

def eval(output,target):
    '''
        result: (batch x step x channel x lat x lon), eg: N x 20 x 5 x H x W
        target: (batch x step x channel x lat x lon), eg: N x 20 x 5 x H x W
    '''
    climates = {
        't2m': 3.1084048748016357,
        'u10': 4.114771819114685,
        'v10': 4.184110546112061,
        'msl': 729.5839385986328,
        'tp': 0.49046186606089276,
    }

    sequence = {
        '1': 3,
        '3': 11,
        '5': 19,
    }
    
    result = {}
    for cid, (name, clim) in enumerate(climates.items()):
        res = []
        for (s_name,sid) in sequence.items():
            out = output[:, sid, cid,:,:]
            tgt = target[:, sid, cid,:,:]
            rmse = compute_rmse(out, tgt)
            # nrmse = (rmse - clim) / clim
            # res.append(rmse)
            # score = np.mean(res)
            result[name+'-'+s_name] = float(rmse)
            # normalized rmse, lower is better,
            # 0 means equal to climate baseline, 
            # less than 0 means better than climate baseline,   
            # -1 means perfect prediction            

        

    # score = np.mean(list(result.values()))
    # result['score'] = float(score)
    return result

def validate(train_param,model, validation_dataloader,wandb=None, inverse_transform_target=None, step=1,
             deepspeed_config=None):
    batch_loss = []
    batch_score = []

    variables_names = ['z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400', 'z500', 'z600', 'z700', 'z850', 'z925', 'z1000', 't50', 't100', 't150', 't200', 't250', 't300', 't400', 't500', 't600', 't700', 't850', 't925', 't1000', 'u50', 'u100', 'u150', 'u200', 'u250', 'u300', 'u400', 'u500', 'u600', 'u700', 'u850', 'u925', 'u1000', 'v50', 'v100', 'v150', 'v200', 'v250', 'v300', 'v400', 'v500', 'v600', 'v700', 'v850', 'v925', 'v1000', 'r50', 'r100', 'r150', 'r200', 'r250', 'r300', 'r400', 'r500', 'r600', 'r700', 'r850', 'r925', 'r1000', 't2m', 'u10', 'v10', 'msl', 'tp']

    batch_loss_param = {

    }
    for v in variables_names:
        batch_loss_param.setdefault(v,[])

    batch_loss = []

    with torch.no_grad():
        model.eval()
        for i,(input,target) in enumerate(tqdm.tqdm(validation_dataloader)):
            # Send input and output data to the GPU/CPU
            input = input.to(train_param.device)
            target = target.to(train_param.device)
            B, in_seq, C_in, H, W = input.shape
            B, out_seq, C_out, H, W = target.shape
            input = input.view(B, in_seq*C_in, H, W)
            target = target.view(B, out_seq*C_out, H, W)
            if step == 1:
                outputs = model(input)
            else:
                # raise NotImplementedError
                outputs = []
                input_ = input
                for i in range(step):
                    tmp = model(input_)
                    outputs.append(tmp[:, -5:, :, :].clone().cpu())
                    input_ = torch.cat([input_[:, C_in:, :, :], tmp], dim=1)
                    
                outputs = torch.cat(outputs, dim=1).to(train_param.device)
            loss = F.mse_loss(input=outputs, target=target, reduction='none')

            batch_loss.append(loss.mean().cpu().item())
            for cid, name in enumerate(variables_names):
                batch_loss_param[name].append(loss[:, cid, :, :].mean().cpu().item())
            
            # Log the loss to wandb
            
            outputs =outputs.view(B, out_seq, C_out, H, W)
            target = target.view(B, out_seq, C_out, H, W)
            
            if inverse_transform_target is not None:
                outputs = inverse_transform_target(outputs)
                target = inverse_transform_target(target)
            
            if step==20:
                eval_output = outputs.cpu().detach()
                eval_target = target.cpu().detach()
                
                score = eval(eval_output, eval_target)
                batch_score.append(score)
            
    if deepspeed_config.local_rank==0 and wandb is not None:
        if step == 1:
            final_loss = np.mean(batch_loss)
            wandb.log({'val_loss': final_loss})
            for name in variables_names:
                batch_loss_param[name]= np.mean(batch_loss_param[name])
            df = pd.DataFrame(batch_loss_param,index=[0],columns=variables_names)
            wandb.log({"valid_table":wandb.Table(data=df)})
        
        if step==20:    
            final_score = {
                k : np.mean([s[k] for s in batch_score])
                for k in batch_score[0].keys()    
            }
            wandb.log(final_score)
            final_loss = np.mean(batch_loss)
            wandb.log({'val_loss_step20': final_loss})
        
        
    return np.mean(batch_loss)
def train(train_param, model, train_set, valid_set=None,valid_set_20step=None,wandb=None, inverse_transform_target=None,
          deepspeed_config=None):
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
        scheduler = NOAMLR(optimizer, warmup_steps=train_param.optimizer.scheduler.warmup_steps, scale=1, model_size=train_param.optimizer.scheduler.hidden_dim)
    else :
        scheduler = None
    
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
            # Calculate the derivatives
            model_engine.backward(loss)
            # Update the model parameters
            model_engine.step()
            
            step += 1
            scheduler.step()        
            if wandb is not None and deepspeed_config.local_rank == 0:
                wandb.log({'loss': loss.item()})
                wandb.log({'lr': optimizer.param_groups[0]['lr']})
            
            if step % train_param.validate_step == 0:
                
                client_sd['step'] = step
                ckpt_id = f'step_{step}loss_{loss.item()}'
                model_engine.save_checkpoint(train_param.save_file, ckpt_id)


        epoch_cost = np.mean(batch_cost)

        # Validation tests
        # if valid_set is not None:
        #     valid_loss=validate(train_param,model, valid_set,wandb=wandb,
        #                         inverse_transform_target=inverse_transform_target,
        #                         deepspeed_config=deepspeed_config)
        #     validation_costs.append(valid_loss)

        # Print progress
        print("Average epoch training cost: ", epoch_cost)
        # if validation_dataloader is not None:
        #     print('Average validation cost:     ', valid_loss)
        print("Epoch time:                   %f seconds" % (time.time() - t_start))
        print("Estimated time to complete:   %.2f minutes, (%.2f seconds)" %
              ((train_param.n_epochs - epoch - 1) * (time.time() - t_start) / 60,
               (train_param.n_epochs - epoch - 1) * (time.time() - t_start)))

        # Save a model checkpoint
        best_result = False
        

    return training_costs, validation_costs
