import os
import numpy as np
import time


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from . import load_dataset

year_len = (np.array([ 366 if i%4 == 0  else 365  for i in range(2007, 2018)]) )*4  # 2007 - 2017
year_cum = np.cumsum(year_len)
year_dic = { i + 1: year_cum[i-2007]  for i in range(2007, 2018)}
year_dic[2007] = 0
# year_dic the start id of one year 

class WeatherDataet_npy(Dataset):

    def __init__(self, data,range , transform=None, target_transform=None, autoregressive = False, preload = False):        
        start = year_dic[range[0]]
        end = year_dic[range[1]]
        assert start < end
        assert start >= 0
        assert end <= 14612
        # left inclusive, right exclusive

        self.transform = transform
        self.target_transform = target_transform
        self.start = start
        self.end = end
        
        self.num_step = 1 if autoregressive else 20  #  default: 20, for 5-days
        
        self.num_data = (end  - start ) - (self.num_step + 1)

        if not autoregressive:
            self.target_range  = slice(65,70)
        else:
            self.target_range  = slice(0,70)

        if preload:
            self.data = np.array(data[start: end, ...])
            self.start = 0
        else:
            self.data = data
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        assert idx < self.num_data
        id = idx + self.start
        # you can reduce it for auto-regressive training 
        
        input = self.data[id : id + 2, : , :, :]
        target = self.data[id + 2 : id + 2 + self.num_step, self.target_range , : , : ]
        
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        
        input = torch.nan_to_num(input) # t c h w 
        target = torch.nan_to_num(target) # t c h w 
        if self.target_transform:
            target = self.target_transform(target)
        if self.transform:
            input = self.transform(input)
        return input, target
    
        
def split_dataset_npy(data_path , npy_name, transform = None, target_transform=None, autoregressive = False,preload_to_memory = False,train_range=(2007,2016), val_range=(2016,2017),**kwargs):
    npy_path = os.path.join(data_path, npy_name)
    if not os.path.exists(npy_path):
        # raise
        print('convert xarray data to npy')
        data = np.memmap(npy_path, dtype = 'float32',mode = 'w+', shape = (14612, 70, 161, 161) , order = 'C')
        time_accumulate = 0
        year_dic[2007]=0
        for q in range(2007, 2017):
            a = load_dataset(data_path, q, q+ 1).x
            data[time_accumulate: time_accumulate + len(a.time.values), :, :, :] = a.values
            time_accumulate += len(a.time.values)
            year_dic[q+1]= time_accumulate
        data.flush()
    else:
        data = np.memmap(npy_path, dtype = 'float32',mode = 'r', shape = (14612, 70, 161, 161) , order = 'C') 
    
    # if preload_to_memory:
    #     data = np.array(data)
    train = WeatherDataet_npy(data, train_range, transform , target_transform , autoregressive , preload= preload_to_memory)
    valid = WeatherDataet_npy(data, val_range, transform , target_transform , autoregressive, preload= preload_to_memory )
    valid_20step = WeatherDataet_npy(data, val_range, transform , target_transform , autoregressive = False, preload= preload_to_memory )
    return train, valid , valid_20step


