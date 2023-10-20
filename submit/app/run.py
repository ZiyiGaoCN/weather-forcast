from model.swin_unet import SwinTransformerSys
import hydra
import torch
import numpy as np
import shutil

# def test_dataset():

#     data_path = '/tcdata/input'
#     # data_path= '/localdata_ssd/gaoziyi/dataset_old/weather_round1_test/input'
#     datas = []
#     for i in range(300):
#         str_i = str(i).zfill(3)
#         data_i = torch.load(f'{data_path}/{str_i}.pt')
#         datas.append(data_i.unsqueeze(0))
#     data = torch.concat(datas, axis=0)
#     # data_torch = torch.from_numpy(data).float()
#     return data

@hydra.main(version_base=None, config_path='./', config_name="swin_transformer")
def main(cfg):
    model = SwinTransformerSys(**cfg.model.param)
    state_dict = torch.load('./save.pt',map_location=cfg.train.device)
    model.load_state_dict(state_dict['model_state_dict'])
    # print(model.keys())
    model.eval()
    model.cuda()
    
    batch_size = 20
    for i in range(700//batch_size):
        datas = []
        for j in range(batch_size):
            id = i*batch_size+j
            str_id = str(id).zfill(3)
            
            input = torch.load(f'/tcdata/input/{str_id}.pt')
            datas += [input.unsqueeze(0)]
            # s = target[j,:,:,:].clone()
        input = torch.cat(datas, axis=0).cuda()
        B, seq_in, C, H, W = input.shape
        input = input.view(B,seq_in*C, H, W)
        
        outputs = []

        for j in range(20):
            checkpoint  = torch.load(f'./finetune_cpkt/save_{j}.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            output = model(input)
            outputs.append(output[:,-5:,...].detach().cpu())
            input_ = torch.cat([input[:,-70:,...],output] , dim=1)
            input = input_
        output = torch.cat(outputs,dim=1)
        for j in range(output.shape[0]):
            str_id = str(i*batch_size+j).zfill(3)
            s = output[j,:,:,:].clone()
            torch.save(s, f'output/{str_id}.pt')    
    import os 
    os.system('zip -r output.zip output')


if __name__ == '__main__':
    main()