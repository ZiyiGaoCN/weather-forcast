from model.swin_unet import SwinTransformerSys
import hydra
import torch
import numpy as np
import shutil
import os
import tqdm

@hydra.main(version_base=None, config_path='./', config_name="swin_transformer")
def main(cfg):
    model = SwinTransformerSys(**cfg.model.param)
    # state_dict = torch.load('./save.pt',map_location=cfg.train.device)
    # model.load_state_dict(state_dict['model_state_dict'])
    # print(model.keys())
    model.eval()
    # model.cuda()
    
    length = 700

    batch_size = 7


    # os.makedirs('/app/data',exist_ok=True)

    data1 = np.memmap('/app/a.npy', dtype='float32', mode='w+', shape=(length,2,70,161,161))
    data2 = np.memmap('/app/b.npy', dtype='float32', mode='w+', shape=(length,2,70,161,161))
    answer = np.memmap('/app/answer.npy', dtype='float32', mode='w+', shape=(length,20,5,161,161))

    for i in range(length):
        str_id = str(i).zfill(3)
        data = torch.load(f'/tcdata/input/{str_id}.pt')
        # data = torch.randn((2,70,161,161))
        data1[i,:,:,:,:] = data[:,:,:,:].numpy()



    for step in range(1,21):
        checkpoint = torch.load(f'./checkpoint/save_{step}.pt',map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        model.eval()
        with torch.no_grad():
            for i in tqdm.tqdm(range(length//batch_size)):
                input = np.array(data1[i*batch_size:(i+1)*batch_size,:,:,:,:])
                input = torch.from_numpy(input).cuda()

                B, seq_in, C, H, W = input.shape
                input = input.view(B,seq_in*C, H, W)
                
                output = model(input)
                output = output.view(B,70,H,W).detach().cpu().numpy()
                input = input.detach().cpu().numpy().reshape(B,seq_in,C,H,W)
                
                answer[i*batch_size:(i+1)*batch_size,step-1,:,:,:] = output[:,65:70,:,:]

                output = output.reshape(B,1,70,H,W)

                new_input = np.concatenate([input[:,1:,:,:,:],output],axis=1)        
                data2[i*batch_size:(i+1)*batch_size,:,:,:,:] = new_input
        data1, data2 = data2, data1
            
    for i in range(length):
        str_id = str(i).zfill(3)
        s = torch.from_numpy(np.array(answer[i,:,:,:,:])).clone()
        torch.save(s, f'output/{str_id}.pt')    
    import os 
    os.system('zip -r output.zip output')


if __name__ == '__main__':
    main()