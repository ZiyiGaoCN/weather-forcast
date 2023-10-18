import os
import numpy as np
import time


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from . import load_dataset

year_len = (np.array([ 366 if i%4 == 0  else 365  for i in range(2007, 2018)]) )*4  # 2007 - 2017
year_cum = np.cumsum(year_len)
year_dic = { i + 1: year_cum[i-2007]  for i in range(2007, 2018)}
year_dic[2007] = 0

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



class WeatherDataet_differentdata(Dataset):
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
         

        if autoregressive, the output channel is 70, otherwise only last 5
        """

        
        self.target = target_data
        if input_data1 is  None:
            self.input1 = self.input2 = self.target
        else:
            self.input1 = input_data1
            if input_data2 is  None:
                self.input2 = self.input1
            else:
                self.input2 = input_data2

        # align the timestanps
        len1, len2, len3 = self.input1.shape[0], self.input2.shape[0], self.target.shape[0]
        assert len3 >= len1 and len3 >= len2 
        assert range[0] < range[1] and range[0] >= 0 and range[1] <= len3
        self.lens = (len1, len2, len3)
        self.start = (range[0] + len1 - len3 -2, range[0] +  len1 - len3 -1,  range[0] )
        self.end = (range[1] + len1 - len3 -2, range[1] +  len1 - len3 -1,  range[1] )
        assert self.start[0] >= 0 and self.start[1] >= 0
        

        if preload:
            self.target = np.array(self.target[self.start[2]: self.end[2], ...])
            if input_data1 is  None:
                self.input1 = self.input2 = self.target
            else:
                self.input1 = np.array(self.input1[self.start[0]: self.end[0], ...])
                if input_data2 is  None:
                    self.input2 = self.input1
                else:
                    self.input2 = np.array(self.input2[self.start[1]: self.end[1], ...])
            self.start = (0,0,0)
            # self.end will not used below
            # self.end = (range[1] - range[0], range[1] - range[0], range[1] - range[0])

        self.num_step = target_step
        self.target_slice = slice(0,70) if autoregressive else slice(65,70)
        
        self.num_data = (range[1]  - range[0] ) - (self.num_step - 1)

        self.target_transform = target_transform
        self.transform = transform
            
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        assert idx < self.num_data
        
        # you can reduce it for auto-regressive training 
        input1 = self.input1[idx + self.start[0]: idx + self.start[0] + 1 ,...]
        input2 = self.input2[idx + self.start[1]: idx + self.start[1] + 1 ,...]
        target = self.target[idx + self.start[2]: idx + self.start[2] + self.num_step, self.target_slice, ...]

        input = np.concatenate((input1, input2), axis= 0)
        
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        
        if self.target_transform:
            target = self.target_transform(target)
        if self.transform:
            input = self.transform(input)
        return input, target

def get_range(year_range = (2007,2009)):
    year_len = (np.array([ 366 if i%4 == 0  else 365  for i in range(2007, 2018)]) )*4  # 2007 - 2017
    year_cum = np.cumsum(year_len)
    year_dic = { i + 1: year_cum[i-2007]  for i in range(2007, 2018)}
    year_dic[2007] = 0
    # year_dic the start id of one year 
    return (year_dic[year_range[0]] + 2, year_dic[year_range[1]])
        
def split_dataset_npy(data_path , npy_name, autoregressive = False,preload_to_memory = False,train_range=(2007,2016), val_range=(2016,2017), npy_len = (-1, -1, -1), **kwargs):

    data = {}
    for i in range(3):
        if  npy_name[i] is not None:
            file = os.path.join(data_path, npy_name[i])
            if not os.path.exists(file):
                raise Exception(f"Please put your npy file at {file}")
            else:
                data[i] = np.memmap(file, dtype = 'float32',mode = 'c', shape = (npy_len[i], 70, 161, 161) , order = 'C') 
        else:
            data[i] = None
    # if preload_to_memory:
    #     data = np.array(data)
    # train = WeatherDataet_npy(data, train_range, autoregressive=autoregressive , preload= preload_to_memory)
    # valid = WeatherDataet_npy(data, val_range, autoregressive=autoregressive, preload= preload_to_memory )
    # valid_20step = WeatherDataet_npy(data, val_range, autoregressive=autoregressive = False, preload= preload_to_memory )
    train_range_stamp = get_range(train_range)
    val_range_stamp = get_range(val_range)
    train = WeatherDataet_differentdata(data[0], data[1], data[2], autoregressive= autoregressive, preload=preload_to_memory, range= train_range_stamp, )
    valid = WeatherDataet_differentdata(data[0], data[1], data[2], autoregressive= autoregressive, preload=preload_to_memory, range= val_range_stamp, )
    return train, valid , 
    # valid_20step


if __name__ == '__main__':
    # a1 = np.ones((10, 7, 5, 5))
    # a2 = np.ones((8, 7, 5, 5))
    # a3 = np.ones((12, 7, 5, 5))

    # data = WeatherDataet_differentdata(a3, a1, a2)
    # print(data[0][0].shape,data[0][1].shape)

    # a, b = split_dataset_npy('.', npy_name= ('data_correct.npy', None, None), autoregressive= True, train_range=(2007,2008), val_range=(2008, 2009), npy_len= (7304, -1,-1))
    pass

