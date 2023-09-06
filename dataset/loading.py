import os
import numpy as np
import pandas as pd
import xarray as xr

import torch
torch.random.seed()
np.random.seed(0)


# data_dir =  # change to you dataset dir

def chunk_time(ds):
    dims = {k:v for k, v in ds.dims.items()}
    dims['time'] = 1
    ds = ds.chunk(dims)
    return ds


def load_dataset_train(data_dir):
    ds = []
    for y in range(2007, 2011):
        data_name = os.path.join(data_dir, f'weather_round1_train_{y}')
        x = xr.open_zarr(data_name, consolidated=True)
        print(f'{data_name}, {x.time.values[0]} ~ {x.time.values[-1]}')
        ds.append(x)
    ds = xr.concat(ds, 'time')
    ds = chunk_time(ds)
    return ds

def load_dataset_valid(data_dir):
    ds = []
    for y in range(2011, 2012):
        data_name = os.path.join(data_dir, f'weather_round1_train_{y}')
        x = xr.open_zarr(data_name, consolidated=True)
        print(f'{data_name}, {x.time.values[0]} ~ {x.time.values[-1]}')
        ds.append(x)
    ds = xr.concat(ds, 'time')
    ds = chunk_time(ds)
    return ds

def loading(data_dir):
    ds = load_dataset_train(data_dir).x
    ds_valid = load_dataset_valid(data_dir).x

    num_step = 20 # for 5-days
    shape = ds.shape # batch x channel x lat x lon 
    times = ds.time.values
    init_times = times[slice(1, -num_step)] 
    num_data = len(init_times)
    names = list(ds.channel.values)
    test_names = names[-5:]

    print(f'\n shape: {shape}')
    print('\n times: {} ~ {}'.format(times[0], times[-1]))
    print('\n init_times: {} ~ {}'.format(init_times[0], init_times[-1]))
    print(f'\n names: {names}')
    print(f'\n test_names: {test_names}\n') 
    
    return ds,ds_valid

if __name__ == '__main__':
    load_dataset('/localdata_ssd/gaoziyi/dataset')
       