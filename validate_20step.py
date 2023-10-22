import torch
from dataset.dataset_npy import WeatherDataet_npy
import numpy as np
from torch.utils.data import DataLoader
import hydra 
import importlib
import torch.nn.functional as F
import tqdm
@hydra.main(version_base=None, config_path='./cfgs/validate', config_name="validate")
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
        dataloader = DataLoader(valid, batch_size=64, shuffle=False, num_workers=8)
    
     

    losses = []

    sum_up = []

    if cfg.validate.step ==1:
        checkpoint =torch.load(cfg.validate.checkpoint,map_location='cpu')
        model.load_state_dict(checkpoint['module'])
        model = model.to(device)
        with torch.no_grad():
            for i,(input,target) in tqdm.tqdm(enumerate(dataloader)):
                outputs = []
                input = input.reshape(-1,140,161,161)
                input = input.to(device)
                output = model(input).detach().cpu()
                target = target.reshape(-1,70,161,161)
                valid_loss = F.mse_loss(output,target,reduction='none')
                sum_up.append(valid_loss.mean())
                losses.append(valid_loss.mean(dim=(0,2,3)))
        print(np.mean(sum_up))        
        print(np.mean(losses,axis=0)[65:70])
    else:
        length = len(valid_20step)

        data1 = np.memmap('/dev/shm/store/validate/a.npy', dtype='float32', mode='w+', shape=(length,2,70,161,161))
        data2 = np.memmap('/dev/shm/store/validate/b.npy', dtype='float32', mode='w+', shape=(length,2,70,161,161))
        answer = np.memmap('/dev/shm/store/validate/answer.npy', dtype='float32', mode='w+', shape=(length,20,5,161,161))
        target = np.memmap('/dev/shm/store/validate/target.npy', dtype='float32', mode='w+', shape=(length,20,5,161,161))

        for i in range(length):
            input, target_ = valid_20step[i]
            data1[i,:,:,:,:] = input[:,:,:,:].numpy()
            target[i,:,:,:,:] = target_[:,:,:,:].numpy()

        batch_size = 64
        Loss_sum = []
        for step in range(1,21):
            checkpoint = torch.load(f'submit/app/checkpoint/save_{step}.pt',map_location="cpu")
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            losses = []
            for i in tqdm.tqdm(range(length//batch_size)):
                # for j in range(batch_size):
                #     id = i*batch_size+j
                #     str_id = str(id).zfill(3)
                    
                #     input = torch.load(f'/tcdata/input/{str_id}.pt')
                #     datas += [input.unsqueeze(0)]
                    # s = target[j,:,:,:].clone()
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
        # for i in range(length//batch_size):
        #     ans = np.array(answer[i*batch_size:(i+1)*batch_size,:,:,:,:])
        #     tar = np.array(target[i*batch_size:(i+1)*batch_size,:,:,:,:])
        #     valid_loss = F.mse_loss(torch.from_numpy(ans),torch.from_numpy(tar),reduction='none')
        #     losses.append(valid_loss.mean(dim=(0,1,3,4)))
        
        # print(np.mean(losses,axis=0))


        

    
    


if __name__ == '__main__':
    main()