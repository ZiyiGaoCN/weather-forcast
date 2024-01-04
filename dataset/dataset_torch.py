import os
import numpy as np
import time
import xarray as xr
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from dataset_mem import load_dataset

year_len = (np.array([ 366 if i%4 == 0  else 365  for i in range(2007, 2018)]) )*4  # 2007 - 2017
year_cum = np.cumsum(year_len)
year_dic = { i + 1: year_cum[i-2007]  for i in range(2007, 2018)}
year_dic[2007] = 0
# year_dic the start id of one year 

def chunk_time(ds):
    dims = {k:v for k, v in ds.dims.items()}
    dims['time'] = 1
    print(f'chunking dims: {dims}')
    ds = ds.chunk(dims)
    return ds

def load_dataset(data_dir, from_year, to_year):
    """from_year: included, to_year: excluded"""
    ds = []
    for y in range(from_year,to_year):
        data_name = os.path.join(data_dir, f'weather_round_train_{y}')
        # print(f'loading {data_name}')
        x = xr.open_zarr(data_name, consolidated=True)
        # print(x.time.values[0:9])
        print(f'{data_name}, {x.time.values[0]} ~ {x.time.values[-1]}')
        # ds.append(x)
        return x
    # ds = xr.concat(ds, 'time')
    # ds = chunk_time(ds)
    return ds


class WeatherDataet_numpy(Dataset):

    def __init__(self, data, range , 
                 transform=None, target_transform=None, step = 1,
                 input_step=1,
                 channul_range=(0,112), preload = False):        
        start = range[0]
        end = range[1]
        assert start < end
        # assert start >= 0
        # assert end <= 14612
        # left inclusive, right exclusive

        self.transform = transform
        self.target_transform = target_transform
        self.start = start
        self.end = end
        
        self.input_step = input_step
        self.num_step = step    #  default: 20, for 5-days
        
        self.num_data = (end  - start ) - (self.num_step) - (self.input_step - 1)

        self.channel_range = slice(channul_range[0], channul_range[1])

        if preload:
            self.data = data.clone()
            # self.start = 0
        else:
            self.data = data
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        assert idx < self.num_data
        id = idx + self.start
        # you can reduce it for auto-regressive training 
        
        input = np.array(self.data[id : id + self.input_step, : , :, :])
        target = np.array(self.data[id + self.input_step : id + self.input_step + self.num_step, self.channel_range , : , : ])
        
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)

        input = torch.nan_to_num(input) # t , in,c h w 
        target = torch.nan_to_num(target) # t, out, c h w 
        if self.target_transform:
            target = self.target_transform(target)
        if self.transform:
            input = self.transform(input)
        return input, target



class WeatherDataet_differentdata_torch(Dataset):
    def __init__(self,target_data, input_data1 = None, input_data2 = None, target_step = 1, autoregressive = True , preload = False, range = (5,10), target_transform = None, transform = None, **kwargs):        
        
        """
        First notice that numpy array in argument are all memmap numpy array. Set preload = True to load to memory

        input_data1 and input_data2 : two npy arrays that correspond to first timestamp and second timestamp
        target_data: the target timestamp, target_data is always the true data, not generated 
        if input_data1 = None, I will use target_data array for input, and input_data2 = None at the same time
        if only input_data2 = None, I will only use input_data1 for the input 

        how to align the timestanps: I will assume the last timestamp of all 3 arrays is the same
        always asume target_data has the longest time range
        the argument range  is for target_data, [start, end)
        target_step is the timestamps of target
        # output target is (end - start) - (target_step -1)
         

        if autoregressive, the output channel is 70, otherwise only 5
        """

        
        self.target = target_data
        self.input1 = input_data1
        self.input2 = input_data2

        # align the timestanps
        len1, len2, len3 = self.input1.shape[0], self.input2.shape[0], self.target.shape[0]
        assert len1 == len2 
        assert len1 == len3 
        assert len1 == 14612

        if preload:
            self.target = self.target.clone()
            self.input1 = self.input1.clone()
            self.input2 = self.input2.clone()
            
        self.num_step = target_step
        self.target_slice = slice(0,70) if autoregressive else slice(65,70)
        self.range = range
        self.num_data = (range[1]  - range[0] ) - (self.num_step - 1) -3

        self.target_transform = target_transform
        self.transform = transform
            
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        assert idx < self.num_data
        
        # you can reduce it for auto-regressive training 
        input1 = self.input1[idx + self.range[0]: idx + self.range[0] + 1 ,...].clone()
        input2 = self.input2[idx + self.range[0] + 1: idx + self.range[0] + 2 ,...].clone()
        target = self.target[idx + self.range[0] + 2: idx + self.range[0] + 2 + self.num_step, self.target_slice, ...].clone()
        input = torch.concat((input1, input2), dim = 0)
        
        if self.target_transform:
            target = self.target_transform(target)
        if self.transform:
            input = self.transform(input)
        return input, target

def create_torchbin(data_path):
    shape = (14612, 70, 161, 161)

    size = np.prod(shape)

    file_name = os.path.join(data_path, 'dataset.bin')
    
    data = torch.from_file(file_name, shared=True, dtype = torch.float16, size = size)

    data = data.view(shape)

    acc = 0
        
    for q in range(2007, 2017):
        print(q)
        a = load_dataset(data_path, q, q + 1).x
        x = torch.from_numpy(a.values).half()
        print(x.shape)
        data[acc: acc + len(a.time.values), :, :, :] = x
        print('range:',acc, acc + len(a.time.values))
        acc += len(a.time.values)
    return data        

if __name__ == '__main__':
    
    # split_dataset_npy('/home/gaoziyi/weather/dataset','/dev/shm/store/checkpoint/original_dataset/dataset.bin')
    create_torchbin('/home/gaoziyi/weather/dataset')
