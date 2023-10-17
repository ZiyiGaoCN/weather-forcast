"""
Implementation of Dataset

the use method is just like __main__ at the end

The data_dir should be like:

data_dir
|
|---- weather_round1_test   
|---- weather_round1_train_2007
|---- weather_round1_train_2008
|---- weather_round1_train_2009
|---- weather_round1_train_2010
|---- weather_round1_train_2011
"""


import os
import pandas as pd
import xarray as xr
import numpy as np
import time

import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
# import matplotlib.patches as patches


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class WeatherDataet(Dataset):
    def __init__(self, data , transform=None, target_transform=None, 
                 normalization_type = 'TAO', is_train = True, data_dir = None):
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.ds = data
        self.values = data.values


        self.shape = self.ds.shape # batch x channel x lat x lon 
        
        self.names = list(self.ds.channel.values)
        self.test_names = self.names[-5:]

        if is_train: 
            if normalization_type == 'TAO': # Time , Lat, Lon 
                assert(data_dir is not None)
                mean = self.ds.mean(dim=['time','lat','lon']).values
                std = self.ds.std(dim=['time','lat','lon']).values
                print(mean.shape)
                normalization_dir = os.path.join(data_dir, 'normalization')
                np.save(os.path.join(normalization_dir, 'mean_TAO.npy'),mean)
                np.save(os.path.join(normalization_dir, 'std_TAO.npy'),std)
            elif normalization_type is None:
                pass
            else:
                raise NotImplementedError
        # self.ds = (self.ds - self.mean) / self.std


    def __len__(self):
        return len(self.ds) - 22 # 22 is the length of target (20) + input (2

    def __getitem__(self, idx):
        # t = self.init_times[idx]
        # t1 = t - pd.Timedelta(hours=6)
        # t2 = t + pd.Timedelta(days=5) # you can reduce it for auto-regressive training 
        # tid = pd.date_range(t1, t2, freq='6h')
        
        # input = self.ds.sel(time=tid[:2]) # you can use subset of input, eg: only surface 
        # target = self.ds.sel(time=tid[2:], channel=self.test_names)
        input = self.values[idx : idx + 2, : , :, :]
        target = self.values[idx + 2 : idx + 22, 65: 70, : , : ]
        
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        
        input = torch.nan_to_num(input) # t c h w 
        target = torch.nan_to_num(target) # t c h w 
        if self.target_transform:
            target = self.target_transform(target)
        if self.transform:
            input = self.transform(input)
        return input, target
    
        
    def visualize(self,time, name):
        
        assert name in self.names
        v = self.ds.sel(time=time, channel=name)

        
        def plot(ds, ax, title):
            ds.plot(
                ax=ax, 
                x='lon', 
                y='lat', 
                transform=ccrs.PlateCarree(),  
                # cbar_kwargs={'label': 'K'},     
                add_colorbar=False
            )
            ax.set_title(title)
            ax.coastlines()
            gl = ax.gridlines(draw_labels=True, linewidth=0.5)
            gl.top_labels = False
            gl.right_labels = False  
            plt.show()  
            

        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})
        plot(v, ax, title=f'{name.upper()}')


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
        ds.append(x)
    ds = xr.concat(ds, 'time')
    ds = chunk_time(ds)
    return ds

def split_dataset(data_dir, ratio=0.8, transform=None, target_transform=None):

    train_data = WeatherDataet(load_dataset(data_dir, 2012, 2016).x, transform, target_transform, data_dir=data_dir,is_train=True)
    test_data = WeatherDataet(load_dataset(data_dir, 2016, 2017).x, transform, target_transform, data_dir=data_dir,is_train=False)
    return train_data, test_data



if __name__ == '__main__':
    # train_data, test_data = split_dataset(data_dir= '../Data', ratio=0.8,  transform=None, target_transform=None )
    import matplotlib.pyplot as plt
    # train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    # a = load_dataset('../Data', 2007, 2012).x
    data = WeatherDataet(load_dataset( '../Data', 2007, 2012).x,)
    a = data[0]




# a = WeatherDataet('./Data')
# a.visualize(time='20080101-00', name='t2m')
# a.visualize(time='20080101-00', name='u10')