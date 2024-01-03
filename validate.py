import torch
import numpy as np
from torch.utils.data import DataLoader
import hydra 
import importlib
import torch.nn.functional as F
import tqdm
import os

from utils import calculate_rmse,calculate_acc
from dataset.dataset_npy import WeatherDataet_npy

def validate_onestep(model,dataloader):
    rmse = []
    acc = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for i,(input,target) in tqdm.tqdm(enumerate(dataloader)):
            input = input.reshape(-1,68,32,64)
            input = input.to(device)
            target = target.to(device)
            
            if model.time_embed:
                time_embed = 1
            else:
                time_embed = None
            
            if model.uncertainty_loss:
                output, sigma = model(input,time_embed)
            else:
                output = model(input,time_embed) #.detach().cpu()
            
            target = target.reshape(-1,68,32,64)
            
            # valid_loss = F.mse_loss(output,target,reduction='none').cpu().float()
            rmse.append(calculate_rmse(output,target))
            acc.append(calculate_acc(output,target))
            
    concat_rmse = np.concatenate(rmse,axis=0)
    concat_acc = np.concatenate(acc,axis=0)
    return np.mean(concat_rmse,axis=0), np.mean(concat_acc,axis=0)

def validate_20step(model,dataloader,step = 20):
    rmses = [ [] for i in range(step)]
    accs = [ [] for i in range(step)]
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for i,(input,target) in tqdm.tqdm(enumerate(dataloader)):
            input = input.reshape(-1,68,32,64)
            input = input.to(device)
            
            for j in range(step):
                if model.time_embed:
                    time_embed = 1
                else:
                    time_embed = None

                if model.uncertainty_loss:
                    output, sigma = model(input,time_embed)
                    # print(sigma.max(),sigma.min())
                else:
                    output = model(input,time_embed) #.detach().cpu()

                # output = model(input,time_embed) #.detach().cpu()
                target_slice = target[:,j,:,:,:] 
                target_slice = target_slice.contiguous().to(device)
                input = output 
                # valid_loss = F.mse_loss(output,target_slice,reduction='none').cpu().float()
                rmses[j].append(calculate_rmse(output,target_slice))
                accs[j].append(calculate_acc(output,target_slice))
    for i in range(step):
        concat_rmse = np.concatenate(rmses[i],axis=0)
        concat_acc = np.concatenate(accs[i],axis=0)
        rmses[i] = np.mean(concat_rmse,axis=0)
        accs[i] = np.mean(concat_acc,axis=0)
        
    return rmses, accs

@hydra.main(version_base=None, config_path='./cfgs/validate', config_name="validate_onestep")
def main(cfg):
    
    # Initialize model 
    model_path = cfg.model.model_path
    model_type = cfg.model.model_type
    model_module = importlib.import_module(model_path)
    model_class = getattr(model_module, model_type)
    model = model_class(**cfg.model.param)
    
    device=cfg.validate.device

    # npy_path = '/local_ssd/gaoziyi/dataset/dataset.npy'
    data = np.memmap(cfg.data.data_path, dtype = 'float32',mode = 'c', shape = (14612, 70, 161, 161) , order = 'C') 
    if cfg.validate.step > 1:
        valid_20step = WeatherDataet_npy(data, (2016,2017), None , None , autoregressive = False, preload= False )
        dataloader = DataLoader(valid_20step, batch_size=64, shuffle=False, num_workers=8)
    else :
        valid = WeatherDataet_npy(data, (2016,2017), None,None , autoregressive= True, preload= False)
        dataloader = DataLoader(valid, batch_size=cfg.validate.batch_size, shuffle=False, num_workers=8)
    
     

    losses = []

    sum_up = []

    if cfg.validate.step ==1:
        checkpoint =torch.load(cfg.validate.checkpoint,map_location='cpu')
        model.load_state_dict(checkpoint['module'])
        model = model.to(device)
        # model.eval()
        # with torch.no_grad():
        #     for i,(input,target) in tqdm.tqdm(enumerate(dataloader)):
        #         outputs = []
        #         input = input.reshape(-1,140,161,161)
        #         input = input.to(device)
        #         output = model(input).detach().cpu()
        #         target = target.reshape(-1,70,161,161)
        #         valid_loss = F.mse_loss(output,target,reduction='none')
        #         sum_up.append(valid_loss.mean())
        #         losses.append(valid_loss.mean(dim=(0,2,3)))
        all_loss, five_param = validate_onestep(model,dataloader)
        print('all_loss:',all_loss)        
        print('five param:',five_param)
    else:
        length = len(valid_20step)

        data1 = np.memmap(os.path.join(cfg.validate.temp_dir,'a.npy'), dtype='float32', mode='w+', shape=(length,2,70,161,161))
        data2 = np.memmap(os.path.join(cfg.validate.temp_dir,'b.npy'), dtype='float32', mode='w+', shape=(length,2,70,161,161))
        answer = np.memmap(os.path.join(cfg.validate.temp_dir,'answer.npy'), dtype='float32', mode='w+', shape=(length,20,5,161,161))
        target = np.memmap(os.path.join(cfg.validate.temp_dir,'target.npy'), dtype='float32', mode='w+', shape=(length,20,5,161,161))

        for i in tqdm.tqdm(range(length)):
            input, target_ = valid_20step[i]
            data1[i,:,:,:,:] = input[:,:,:,:].numpy()
            target[i,:,:,:,:] = target_[:,:,:,:].numpy()
        
        start = 13148

        batch_size = 64
        Loss_sum = []
        with torch.no_grad():
            for step in range(1,21):

                if cfg.validate.same == True:
                    if step ==1:
                        checkpoint = torch.load(os.path.join(cfg.validate.checkpoint,f'save_1.pt'),map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                else:    
                    checkpoint = torch.load(os.path.join(cfg.validate.checkpoint,f'save_{step}.pt'),map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                losses = []
                loss_all_sum = []
                for i in tqdm.tqdm(range(length//batch_size)):
                    input = np.array(data1[i*batch_size:(i+1)*batch_size,:,:,:,:])
                    input = torch.from_numpy(input).to(device)

                    B, seq_in, C, H, W = input.shape
                    input = input.view(B,seq_in*C, H, W)
                    
                    output = model(input)
                    output = output.view(B,70,H,W).detach().cpu().numpy()
                    input = input.detach().cpu().numpy().reshape(B,seq_in,C,H,W)
                    
                    answer[i*batch_size:(i+1)*batch_size,step-1,:,:,:] = output[:,65:70,:,:]
                    
                    tar = torch.from_numpy(np.array(target[i*batch_size:(i+1)*batch_size,step-1,:,:,:]))
                    loss = F.mse_loss(torch.from_numpy(output[:,65:70,:,:]),tar,reduction='none').mean(dim=(0,2,3))
                    losses.append(loss)

                    label = torch.from_numpy(data[i*batch_size+start+step+1:(i+1)*batch_size+start+step+1,:,:,:])
                    loss_all = F.mse_loss(torch.from_numpy(output),label,reduction='mean').item()
                    loss_all_sum.append(loss_all)
                    output = output.reshape(B,1,70,H,W)



                    new_input = np.concatenate([input[:,1:,:,:,:],output],axis=1)        
                    data2[i*batch_size:(i+1)*batch_size,:,:,:,:] = new_input
                data1, data2 = data2, data1
                losses = np.mean(losses,axis=0)
                print(np.mean(loss_all_sum))
                print(losses)
                Loss_sum.append(losses)
            print(np.mean(Loss_sum,axis=0))



        

    
    


if __name__ == '__main__':
    main()