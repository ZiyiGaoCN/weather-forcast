import torch
import numpy as np
from torch.utils.data import DataLoader
import hydra 
import importlib
import torch.nn.functional as F
import tqdm
import os

from dataset.dataset_npy import WeatherDataet_npy

def validate_onestep(model,dataloader):
    losses = []
    sum_up = []
    model.eval()
    device = next(model.parameters()).device
    for i,(input,target) in tqdm.tqdm(enumerate(dataloader)):
        outputs = []
        input = input.reshape(-1,140,161,161)

        input = input.to(device)
        
        input.requires_grad_()

        target = target.to(device)
        output = model(input) #.detach().cpu()
        
        unstale = []
        names = ['z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400', 'z500', 'z600', 'z700', 'z850', 'z925', 'z1000', 't50', 't100', 't150', 't200', 't250', 't300', 't400', 't500', 't600', 't700', 't850', 't925', 't1000', 'u50', 'u100', 'u150', 'u200', 'u250', 'u300', 'u400', 'u500', 'u600', 'u700', 'u850', 'u925', 'u1000', 'v50', 'v100', 'v150', 'v200', 'v250', 'v300', 'v400', 'v500', 'v600', 'v700', 'v850', 'v925', 'v1000', 'r50', 'r100', 'r150', 'r200', 'r250', 'r300', 'r400', 'r500', 'r600', 'r700', 'r850', 'r925', 'r1000', 't2m', 'u10', 'v10', 'msl', 'tp']

        for j in tqdm.tqdm(range(70)):
            gard = torch.autograd.grad(output[0,j,0,0],input,retain_graph=True,create_graph=False)[0]
            print(gard.max())
            print(gard.min())
            print(gard.mean())
            std= gard.std()
            unstale.append((std,names[j]))
        unstale.sort(key=lambda x:x[0],reverse=True)
        print(unstale[:70])
        break 

        # target = target.reshape(-1,70,161,161)
        # valid_loss = F.mse_loss(output,target,reduction='none')
        
        # sum_up.append(valid_loss.mean())
        # losses.append(valid_loss.mean(dim=(0,2,3)))
    # return np.mean(sum_up), np.mean(losses,axis=0)[65:70]


@hydra.main(version_base=None, config_path='./cfgs/validate', config_name="validate_onestep")
def main(cfg):
    
    # Initialize model 
    model_path = cfg.model.model_path
    model_type = cfg.model.model_type
    model_module = importlib.import_module(model_path)
    model_class = getattr(model_module, model_type)
    model = model_class(**cfg.model.param)
    
    device=cfg.validate.device
    print(device)

    # npy_path = '/local_ssd/gaoziyi/dataset/dataset.npy'
    data = np.memmap(cfg.data.data_path, dtype = 'float32',mode = 'c', shape = (14612, 70, 161, 161) , order = 'C') 
    if cfg.validate.step > 1:
        valid_20step = WeatherDataet_npy(data, (2016,2017), None , None , autoregressive = False, preload= False )
        dataloader = DataLoader(valid_20step, batch_size=1, shuffle=False, num_workers=8)
    else :
        valid = WeatherDataet_npy(data, (2016,2017), None,None , autoregressive= True, preload= False)
        dataloader = DataLoader(valid, batch_size=1, shuffle=False, num_workers=8)
        print("finish loading data")
     

    losses = []

    sum_up = []

    if cfg.validate.step ==1:
        checkpoint =torch.load(cfg.validate.checkpoint,map_location=device)
        model.load_state_dict(checkpoint['module'])
        model = model.to(device)
        print("finish loading model")
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

        batch_size = 64
        Loss_sum = []
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

                output = output.reshape(B,1,70,H,W)

                new_input = np.concatenate([input[:,1:,:,:,:],output],axis=1)        
                data2[i*batch_size:(i+1)*batch_size,:,:,:,:] = new_input
            data1, data2 = data2, data1
            losses = np.mean(losses,axis=0)
            print(losses)
            Loss_sum.append(losses)
        print(np.mean(Loss_sum,axis=0))



        

    
    


if __name__ == '__main__':
    main()