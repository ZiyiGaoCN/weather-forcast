import torch
import tqdm
import numpy as np

def Construct_Dataset(model,dataloader, output_path , device="cuda:0"):
    model = model.to(device)
    data = np.memmap(output_path, dtype = 'float32',mode = 'w+', shape = (len(dataloader)-1, 70, 161, 161) , order = 'C')
    id = 0
    for i,(input) in enumerate(tqdm.tqdm(dataloader)):
        # Send input and output data to the GPU/CPU
        input = input.to(device)
        B, in_seq, C_in, H, W = input.shape
        input = input.view(B, in_seq*C_in, H, W)
        output = model(input).cpu().numpy()
        data[id: id + output.shape[0], :, :, :] = output
        id += output.shape[0]
    