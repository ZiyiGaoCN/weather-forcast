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
    
    # test_data = test_dataset()
    # print(test_data.shape)
    # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=cfg.data.batch_size_val, shuffle=False, num_workers=2)
    
    # with torch.no_grad():
    #     for i, data in enumerate(test_dataloader):
    #         data = data.cuda()
    #         print(data.shape)
    #         # data = data.unsqueeze(0)
    #         B, seq_in, C, H, W = data.shape
    #         data = data.view(B,seq_in*C, H, W)
    #         output = model(data)
    #         print(output.shape)

    #         output = model(data)
    #         target = output.view(B, 20, 5, H, W).cpu().detach()
    #         # save target to pt
    #         for j in range(target.shape[0]):
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
        output = model(input.cuda())
        output = output.view(B, 20, 5, H, W).cpu().detach()
        for j in range(output.shape[0]):
            str_id = str(i*batch_size+j).zfill(3)
            s = output[j,:,:,:].clone()
            torch.save(s, f'output/{str_id}.pt')    
    import os 
    os.system('zip -r output.zip output')


if __name__ == '__main__':
    main()