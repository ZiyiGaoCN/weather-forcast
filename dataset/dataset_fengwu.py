from cgitb import small
import os
from venv import logger
import numpy as np
import time
import xarray as xr
import torch
from torch.utils.data import Dataset
import string
import random
import loguru

def random_string(length):
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


class Dataset_Fengwu(Dataset):
    def __init__(self, data , replay_buffer = None, replay_time = None, replay_step = None,
                 replay_sigma = None,
                 map_id = None, buffer_file = None , size = 20000 ,
                 transform=None, target_transform=None,
                 uncertainty_finetune= False ,
                 channul_range=(0,112), preload = False,
                 multi_step = False) -> None:
        super().__init__()
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.num_step = 1   #  default: 20, for 5-days
        
        # self.channel_range = slice(channul_range[0], channul_range[1])

        self.data = data
        self.replay_buffer = replay_buffer    
        self.replay_time = replay_time
        self.replay_sigma = replay_sigma
        self.map_id = map_id
        
        self.size = size
        self.uncertainty_finetune = uncertainty_finetune
        
        self.buffer_dataset_input = np.memmap(buffer_file.replace('.npy','_input.npy'), dtype = 'float32',mode = 'w+', shape = (size, self.replay_buffer.shape[2], self.replay_buffer.shape[3], self.replay_buffer.shape[4]) , order = 'C')
        self.buffer_dataset_target = np.memmap(buffer_file.replace('.npy','_target.npy'), dtype = 'float32',mode = 'w+', shape = (size, self.data.shape[1], self.data.shape[2], self.data.shape[3]) , order = 'C')
        self.buffer_dataset_time = np.memmap(buffer_file.replace('.npy','_time.npy'), dtype = 'int64',mode = 'w+', shape = (size) , order = 'C')
        self.buffer_dataset_step = np.memmap(buffer_file.replace('.npy','_step.npy'), dtype = 'int64',mode = 'w+', shape = (size) , order = 'C')
        self.buffer_dataset_time_step = np.memmap(buffer_file.replace('.npy','_time_embed.npy'), dtype = 'int64',mode = 'w+', shape = (size) , order = 'C')
        self.buffer_dataset_sigma = np.memmap(buffer_file.replace('.npy','_sigma.npy'), dtype = 'int64',mode = 'w+', shape = (size,self.data.shape[1]) , order = 'C')
        current_size = 0
        for i in range(20):
            if i > 0:
                input = self.replay_buffer[i,self.map_id[i],...]
                time = self.replay_time[i,self.map_id[i]]
                sigma = self.replay_sigma[i,self.map_id[i],...]
            else: 
                input = self.data[self.map_id[i],...]
                time = self.map_id[i]
                sigma = np.zeros((input.shape[0],input.shape[1])) 
            if multi_step:
                time_step = np.random.randint(1, min(4,20-i) + 1 ,(input.shape[0],))
            else:
                time_step = np.ones_like(time)
            
            target = self.data[time + time_step,...]
            step = np.ones_like(time)*i
            batch_size = input.shape[0]
            self.buffer_dataset_input[current_size:current_size+batch_size,...] = input
            self.buffer_dataset_target[current_size:current_size+batch_size,...] = target
            self.buffer_dataset_time[current_size:current_size+batch_size] = time
            self.buffer_dataset_step[current_size:current_size+batch_size] = step
            self.buffer_dataset_time_step[current_size:current_size+batch_size] = time_step
            self.buffer_dataset_sigma[current_size:current_size+batch_size] = sigma
            current_size += batch_size
        print(current_size,size)
        assert current_size == size
            # assert self.buffer_dataset[self.max_id[i],...].shape
            
    def __len__(self):
        return self.size
            
    def __getitem__(self, idx):
        id = idx
        # loguru.logger.info("get item: {}".format(id))
        input = torch.from_numpy(np.array(self.buffer_dataset_input[id : id + 1, : , :, :]))
        target = torch.from_numpy(np.array(self.buffer_dataset_target[id : id + 1, : , :, :]))
        time = torch.from_numpy(np.array(self.buffer_dataset_time[id]))
        step = torch.from_numpy(np.array(self.buffer_dataset_step[id]))
        sigma = torch.from_numpy(np.array(self.buffer_dataset_sigma[id: id + 1,:])).view(1,-1,1,1)
        eps = torch.randn_like(input)
        input = input + sigma * eps
        time_embed = torch.from_numpy(np.array(self.buffer_dataset_time_step[id]))       
        return input, target, time, step, time_embed

class Replay_buffer_fengwu():

    def __init__(self, data, train_range , buffer_file = '/dev/shm/store/buffer/',
                 small_buffer_folder = None,
                 buffer_size=58436, shape = (58436, 73, 32, 64),
                 transform=None, target_transform=None, 
                 uncertainty_finetune = False,
                 channul_range=(0,112), preload = False):        
        start = train_range[0]
        end = train_range[1]
        assert start < end
        

        self.transform = transform
        self.target_transform = target_transform
        self.start = start
        self.end = end
        
        self.num_step = 20   #  default: 20, for 5-days
        
        self.num_data = (end  - start ) - (self.num_step)

        self.channel_range = channul_range

        if preload:
            self.data = np.array(data)
        else:
            self.data = data
            
        random_str = random_string(10)
        # buffer_file = buffer_file
        
        self.small_buffer_folder = small_buffer_folder
        
        self.buffer_file = buffer_file
        self.buffer_size = buffer_size
        self.buffer = np.memmap(buffer_file, dtype = 'float32',mode = 'w+', shape = (20, self.buffer_size, shape[1],shape[2],shape[3]) , order = 'C')
        self.buffer_time = np.memmap(buffer_file.replace('.npy','_time.npy'), dtype = 'int64',mode = 'w+', shape = (20, self.buffer_size) , order = 'C')
        self.buffer_sigma = np.memmap(buffer_file.replace('.npy','_sigma.npy'), dtype = 'float32',mode = 'w+', shape = (20, self.buffer_size, shape[1]) , order = 'C')
        # self.buffer_step = np.memmap(buffer_file.replace('.npy','_step.npy'), dtype = 'int64',mode = 'w+', shape = (20, self.buffer_size) , order = 'C')
        
        # self.buffer[:end-start - 1 ,...] = self.data[start : end - 1 , ...]
        # self.buffer_time[:end-start - 1] = np.arange(start+1,end)
        # self.buffer_step[:end-start - 1] = 1       
        # self.idx = end - start - 1
        # self.max_idx = self.idx  
        self.idx = np.zeros(20,dtype = np.int64)
        self.max_idx = np.zeros(20, dtype= np.int64)
        self.idx[0] = end - start - self.num_step
        self.max_idx[0] = end - start - self.num_step
        # self.buffer[0, :end-start - self.num_step ,...] = self.data[start : end - self.num_step , ...]
        # self.buffer_time[0, :end-start - self.num_step] = np.arange(start+1,end - self.num_step + 1)
        # self.buffer_step[0, :end-start - self.num_step] = 1
        
    # def __len__(self):
    #     return self.max_idx

    def add_buffer(self, data, time, step,sigma = None):
        assert step > 0
        if step == 20:
            loguru.logger.info("Skip")
            return 
        batch_size = data.shape[0]
        if self.idx[step] + batch_size <= self.buffer_size:
            slices = [(slice(self.idx[step], self.idx[step] + batch_size),slice(0,batch_size))]
        else:
            slices = [
                (slice(self.idx[step], self.buffer_size),slice(0,self.buffer_size - self.idx[step])),
                (slice(0, batch_size - (self.buffer_size - self.idx[step])),slice(self.buffer_size - self.idx[step], batch_size))
            ]
        # print(slices)
        # print(self.buffer_time.shape,time.shape)
        # print(self.buffer_step.shape,step.shape)
        for sa, sb in slices:
            # print(data[sb, ...].shape)
            # print(self.buffer[sa, ...].shape)
            self.buffer[step , sa, :,:,:] = data[sb, :,:,:]
            self.buffer_time[step,sa] = time[sb]
            
            if sigma is not None:
                self.buffer_sigma[step, sa, ...] = sigma[sb,:, 0,0]
            # self.buffer_step[sa] = step[sb] + 1
        self.max_idx[step] = max(self.max_idx[step], min(self.idx[step] + batch_size,self.buffer_size) )
        self.idx[step] = (self.idx[step] + batch_size) % self.buffer_size
    
    def add_buffer_20step(self, data, time, step, sigma=None):
        pass
        indices = []
        for i in range(20):
            index = np.where(step == i)[0]
            # if index.size > 0:
            if index.size == 0:
                continue
            # indices.append(index)
            self.add_buffer(data[index,...],time[index],i, sigma[index,...] if sigma is not None else None)
    
    def get_id(self,size,step):
        # if self.idx + size < self.max_idx:
        #     potential_idx = np.concatenate([np.arange(self.idx+size, self.max_idx),np.arange(0,self.idx)])
        # else:
        potential_idx = np.arange(0,self.max_idx[step])
        idx = np.random.choice(potential_idx, size = size, replace=False)
        return idx
    
    def ids(self,size,sample_ratio):
        max_ratio = np.zeros(20,dtype=np.float32)
        final_size = np.zeros(20,dtype=np.int64)
        for i in range(20):
            max_ratio[i] = self.max_idx[i]/size
            
        for i in range(20):
            if max_ratio[i] < sample_ratio[i]:
                sample_ratio[i] = max_ratio[i]
        for i in range(20):
            final_size[i] = int(size*sample_ratio[i])
        all = np.sum(final_size)
        final_size[0] += size - all
        
        ids = [
            self.get_id(final_size[i],i) for i in range(20)
        ]
        return ids
                
    
    def build_dataset(self, size = 10000, sample_ratio=None, uncertainty_finetune=False, multi_step = False):
        # ban: [self.idx, self.idx + size]
        
        if sample_ratio is None:
            sample_ratio = np.ones(20)/20
        ids = self.ids(size,sample_ratio)
        
        
        
        return Dataset_Fengwu(data=self.data,
                                replay_buffer = self.buffer, replay_time = self.buffer_time, replay_step=None,
                                replay_sigma = self.buffer_sigma ,
                                map_id = ids, buffer_file= os.path.join(self.small_buffer_folder,'_dataset.npy'),
                                size = size,
                                transform=self.transform, target_transform=self.target_transform, 
                                uncertainty_finetune = uncertainty_finetune,
                                channul_range=self.channel_range, preload = False,
                                multi_step = multi_step)
             

    # def __getitem__(self, idx):
    #     assert idx < self.num_data
    #     id = idx + self.start
    #     # you can reduce it for auto-regressive training 
        
    #     assert id < self.data.shape[0]
        
    #     input = np.array(self.data[id : id + 1, : , :, :])
    #     target = np.array(self.data[id + 1 : id + 1 + self.num_step, self.channel_range , : , : ])
        
    #     input = torch.from_numpy(input)
    #     target = torch.from_numpy(target)

    #     input = torch.nan_to_num(input) # t c h w 
    #     target = torch.nan_to_num(target) # t c h w 
    #     if self.target_transform:
    #         target = self.target_transform(target)
    #     if self.transform:
    #         input = self.transform(input)
    #     return input, target
