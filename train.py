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

def     eval(output,target):
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
    
    result = {}
    for cid, (name, clim) in enumerate(climates.items()):
        res = []
        for sid in range(output.shape[1]):
            out = output[:, sid, cid,:,:]
            tgt = target[:, sid, cid,:,:]
            rmse = compute_rmse(out, tgt)
            # nrmse = (rmse - clim) / clim
            res.append(rmse)
            
            # normalized rmse, lower is better,
            # 0 means equal to climate baseline, 
            # less than 0 means better than climate baseline,   
            # -1 means perfect prediction            

        score = np.mean(res)
        result[name] = float(score)

    score = np.mean(list(result.values()))
    result['score'] = float(score)
    return result

def validate(train_param,model, validation_dataloader,wandb=None, inverse_transform_target=None):
    batch_loss = []
    batch_score = []
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
            outputs = model(input)
            loss = F.mse_loss(input=outputs, target=target)
            batch_loss.append(loss.item())
            # Log the loss to wandb
            
            outputs =outputs.view(B, out_seq, C_out, H, W)
            target = target.view(B, out_seq, C_out, H, W)
            
            if inverse_transform_target is not None:
                outputs = inverse_transform_target(outputs)
                target = inverse_transform_target(target)
            
            eval_output = outputs.cpu().detach()
            eval_target = target.cpu().detach()
            
            score = eval(eval_output, eval_target)
            batch_score.append(score)
            
    if wandb is not None:
        final_loss = np.mean(batch_loss)
        final_score = {
            k : np.mean([s[k] for s in batch_score])
            for k in batch_score[0].keys()    
        }
        wandb.log({'val_loss': final_loss})
        wandb.log(final_score)
        
    return np.mean(batch_loss)
def train(train_param, model, train_dataloader, validation_dataloader=None,wandb=None, inverse_transform_target=None):
    """
    Train the ForecastNet model on a provided dataset. 
    In the following variable descriptions, the input_seq_length is the length of the input sequence 
    (2*seasonal_period in the paper) and output_seq_length is the number of steps-ahead to forecast 
    (seasonal_period in the paper). The n_batches is the total number batches in the dataset. The 
    input_dim and output_dim are the dimensions of the input sequence and output sequence respectively
    (in the paper univariate sequences were used where input_dim=output_dim=1).
    :param fcstnet: A forecastNet object defined by the class in forecastNet.py. 
    :param train_x: Input training data in the form [input_seq_length, n_batches, input_dim]
    :param train_y: Target training data in the form [output_seq_length, n_batches, output_dim]
    :param validation_x: Optional input validation data in the form [input_seq_length, n_batches, input_dim]
    :param validation_y: Optional target validation data in the form [output_seq_length, n_batches, output_dim]
    :param restore_session: If true, restore parameters and keep training, else train from scratch
    :return: training_costs: a list of training costs over the set of epochs
    :return: validation_costs: a list of validation costs over the set of epochs
    """

    # Convert numpy arrays to Torch tensors
    # if type(train_x) is np.ndarray:
    #     train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    # if type(train_y) is np.ndarray:
    #     train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
    # if type(validation_x) is np.ndarray:
    #     validation_x = torch.from_numpy(validation_x).type(torch.FloatTensor)
    # if type(validation_y) is np.ndarray:
    #     validation_y = torch.from_numpy(validation_y).type(torch.FloatTensor)

    # Format inputs
    # train_x = format_input(train_x)
    # validation_x = format_input(validation_x)

    # validation_x = validation_x.to(fcstnet.device)
    # validation_y = validation_y.to(fcstnet.device)

    # train_param.optimizer

    if train_param.optimizer.name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=train_param.optimizer.learning_rate, weight_decay=train_param.optimizer.weight_decay)

    # Initialise model with predefined parameters
    if hasattr(train_param,'restore_ression'):
        # Load model parameters
        checkpoint = torch.load(model.save_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    # Number of batch samples
    # n_samples = train_x.shape[0]

    # List to hold the training costs over each epoch
    training_costs = []
    validation_costs = []

    # Set in training mode
    model.to(train_param.device)
    model.train()

    if validation_dataloader is not None:
        valid_loss=validate(train_param,model, validation_dataloader,wandb=wandb,inverse_transform_target=inverse_transform_target)
        validation_costs.append(valid_loss)

    # Training loop
    for epoch in range(train_param.n_epochs):

        # Start the epoch timer
        t_start = time.time()

        # Print the epoch number
        print('Epoch: %i of %i' % (epoch + 1, train_param.n_epochs))

        # Initial average epoch cost over the sequence
        batch_cost = []

        # Counter for permutation loop
        count = 0

        # Permutation to randomly sample from the dataset
        # permutation = np.random.permutation(np.arange(0, n_samples, fcstnet.batch_size))
        model.train()
        # Loop over the permuted indexes, extract a sample at that index and run it through the model
        for i,(input,target) in enumerate(tqdm.tqdm(train_dataloader)):
            # Send input and output data to the GPU/CPU
            input = input.to(train_param.device)
            target = target.to(train_param.device)

            # Zero the gradients
            optimizer.zero_grad()
            # Compute outputs and loss for a mixture density network output
            
            B, in_seq, C_in, H, W = input.shape
            B, out_seq, C_out, H, W = target.shape
            input = input.view(B, in_seq*C_in, H, W)
            target = target.view(B, out_seq*C_out, H, W)
            outputs = model(input)
            loss = F.mse_loss(input=outputs, target=target)
            batch_cost.append(loss.item())
            # Calculate the derivatives
            loss.backward()
            # Update the model parameters
            optimizer.step()
            
            # Log the loss to wandb
            if wandb is not None:
                wandb.log({'loss': loss.item()})

            if count % 50 == 0:
                print("Average cost after training batch %i of %i: %f" % (count, input.shape[0], loss.item()))
            count += 1
        # Find average cost over sequences and batches
        epoch_cost = np.mean(batch_cost)
        # Calculate the average training cost over the sequence
        training_costs.append(epoch_cost)

        # Validation tests
        if validation_dataloader is not None:
            valid_loss=validate(train_param,model, validation_dataloader,wandb=wandb,inverse_transform_target=inverse_transform_target)
            validation_costs.append(valid_loss)
        # Print progress
        print("Average epoch training cost: ", epoch_cost)
        if validation_dataloader is not None:
            print('Average validation cost:     ', valid_loss)
        print("Epoch time:                   %f seconds" % (time.time() - t_start))
        print("Estimated time to complete:   %.2f minutes, (%.2f seconds)" %
              ((train_param.n_epochs - epoch - 1) * (time.time() - t_start) / 60,
               (train_param.n_epochs - epoch - 1) * (time.time() - t_start)))

        # Save a model checkpoint
        best_result = False
        if validation_dataloader is None:
            if training_costs[-1] == min(training_costs):
                best_result = True
        else:
            if validation_costs[-1] == min(validation_costs):
                best_result = True
        if best_result or (epoch+1) % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(train_param.save_file,f'save_{epoch}.pt'))
            print("Model saved in path: %s" % train_param.save_file)

    return training_costs, validation_costs
