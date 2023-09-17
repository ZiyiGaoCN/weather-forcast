import os
import numpy as np
import time


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class WeatherDataet_npy(Dataset):

    def __init__(self, data, start = 0, end = 7304, transform=None, target_transform=None):
        

        assert start < end
        assert start > 0
        assert end <= 7304
        # left inclusive, right exclusive

        self.transform = transform
        self.target_transform = target_transform
        self.start = start
        self.end = end
        
        self.data = data
        self.num_step = 20 #  default: 20, for 5-days
        
        
        self.num_data = (end  - start ) - (self.num_step + 1)
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        assert idx < self.num_data
        id = idx + self.start
        # you can reduce it for auto-regressive training 
        
        input = self.data[id : id + 2, : , :, :]
        target = self.data[id + 2 : id + 22, 65: 70, : , : ]
        
        input = torch.from_numpy(input.values)
        target = torch.from_numpy(target.values)
        
        input = torch.nan_to_num(input) # t c h w 
        target = torch.nan_to_num(target) # t c h w 
        if self.target_transform:
            target = self.target_transform(target)
        if self.transform:
            input = self.transform(input)
        return input, target
    
        
def split_dataset_npy(npy_dir, transform = None, target_transform=None ):
    data = np.memmap(os.path.join(npy_dir, 'data_normailze.npy'), dtype = 'float32',mode = 'r', shape = (7304, 70, 161, 161) , order = 'C') 
    train = WeatherDataet_npy(data, 0, 5844, transform , target_transform )
    valid = WeatherDataet_npy(data, 5844, 7304, transform , target_transform )
    return train, valid


