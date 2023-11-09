import torch
import tqdm
import numpy as np

def Construct_Dataset(model, dataloader , data , device="cuda:0", start = 0):
    model = model.cuda()
    id = start
    with torch.no_grad():
        model.eval()
        for i,(input,target) in enumerate(tqdm.tqdm(dataloader)):
            # Send input and output data to the GPU/CPU
            input = input.cuda()
            B, in_seq, C_in, H, W = input.shape
            input = input.view(B, in_seq*C_in, H, W)
            output = model(input).detach().cpu().numpy()
            data[id: id + output.shape[0], :, :, :] = output
            id += output.shape[0]
            # del input
    