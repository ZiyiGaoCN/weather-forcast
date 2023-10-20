import torch
from dataset.dataset_npy import WeatherDataet_npy
import numpy as np
from torch.utils.data import DataLoader
import hydra 
import importlib
import torch.nn.functional as F
import tqdm
@hydra.main(version_base=None, config_path='./cfgs/finetune', config_name="finetune")
def main(cfg):
    
    # Initialize model 
    model_path = cfg.model.model_path
    model_type = cfg.model.model_type
    model_module = importlib.import_module(model_path)
    model_class = getattr(model_module, model_type)
    model = model_class(**cfg.model.param)
    
    device="cuda:7"

    npy_path = '/local_ssd/gaoziyi/dataset/dataset.npy'
    data = np.memmap(npy_path, dtype = 'float32',mode = 'c', shape = (14612, 70, 161, 161) , order = 'C') 
    valid_20step = WeatherDataet_npy(data, (2016,2017), None , None , autoregressive = False, preload= False )
    dataloader = DataLoader(valid_20step, batch_size=64, shuffle=False, num_workers=8)
    
    losses = []

    sum_up = []

    with torch.no_grad():
        for i,(input,target) in tqdm.tqdm(enumerate(dataloader)):
            outputs = []
            input = input.reshape(-1,140,161,161)
            input = input.to(device)
            
            for j in range(20):
                checkpoint  = torch.load(f'/local_ssd/gaoziyi/finetune_ckpt/checkpoint/save_{j+1}.pt')
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)
                model.eval()
                output = model(input)
                outputs.append(output[:,-5:,...].detach().cpu())
                input_0 , input_1 = torch.split(input,70,dim=1)
                input_ = torch.cat([input_1[:,-70:,...],output] , dim=1) 
                del input_0
                # input_ = torch.cat([input[:,-70:,...],output] , dim=1)
                input = input_
            output = torch.cat(outputs,dim=1)
            output = output.reshape(-1,100,161,161)
            target = target.reshape(-1,100,161,161)
            valid_loss = F.mse_loss(output,target,reduction='none')
            sum_up.append(valid_loss.mean())
            losses.append(valid_loss.mean(dim=(0,2,3)))
    print(np.mean(sum_up))        
    print(np.mean(losses,axis=0).reshape(20,5))


        

    
    


if __name__ == '__main__':
    main()