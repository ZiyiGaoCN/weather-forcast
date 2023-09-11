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

import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
# import matplotlib.patches as patches

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class WeatherDataet(Dataset):
    def __init__(self, data , transform=None, target_transform=None, num_step = 20, test=False):
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.ds = data


        self.num_step = num_step #  default: 20, for 5-days
        self.shape = self.ds.shape # batch x channel x lat x lon 
        self.times = self.ds.time.values
        self.init_times = self.times[slice(1, -self.num_step)] 
        self.num_data = len(self.init_times)
        self.names = list(self.ds.channel.values)
        self.test_names = self.names[-5:]




    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        assert idx < self.num_data
        t = self.init_times[idx]
        t1 = t - pd.Timedelta(hours=6)
        t2 = t + pd.Timedelta(days=5) # you can reduce it for auto-regressive training 
        tid = pd.date_range(t1, t2, freq='6h')
        
        input = self.ds.sel(time=tid[:2]) # you can use subset of input, eg: only surface 
        target = self.ds.sel(time=tid[2:], channel=self.test_names)
        
        input = torch.from_numpy(input.values)
        target = torch.from_numpy(target.values)
        
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
        
# class WeatherTest(Dataset):
#     def __init__(self) -> None:
    
        pass
def split_dataset(data_dir, ratio=0.8, transform=None, target_transform=None):

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
            data_name = os.path.join(data_dir, f'weather_round1_train_{y}')
            # print(f'loading {data_name}')
            x = xr.open_zarr(data_name, consolidated=True)
            # print(x.time.values[0:9])
            print(f'{data_name}, {x.time.values[0]} ~ {x.time.values[-1]}')
            ds.append(x)
        ds = xr.concat(ds, 'time')
        ds = chunk_time(ds)
        return ds

    train_data = WeatherDataet(load_dataset(data_dir, 2007, 2011).x, transform, target_transform)
    test_data = WeatherDataet(load_dataset(data_dir, 2011, 2012).x, transform, target_transform)
    return train_data, test_data

def load_dataset_test(data_dir):
    import os 
    test_dir = os.path.join(data_dir,'weather_round1_test/input')
    lists = os.listdir(test_dir)
    lists = sorted(lists)
    inputs = []
    for file in lists:
        input = torch.load(os.path.join(test_dir,file))
        inputs.append(input)
    inputs = torch.stack(inputs)
    print(inputs.shape)
    return inputs

def test_dataset(data_dir, transform=None, target_transform=None):
    return load_dataset_test(data_dir)

    # total = WeatherDataet(data_dir, transform, target_transform)
    # train_size = int(len(total) * ratio)
    # test_size = len(total) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(total, [train_size, test_size])
    # return train_dataset, test_dataset




if __name__ == '__main__':
    pass
    # test_dataset = WeatherDataet(test_data, transform=None, target_transform=None,test=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # for i, x in enumerate(test_dataloader):
    #     print(i, x.shape)
    #     break
    # train_data, test_data = split_dataset(data_dir= '../../dataset', ratio=0.8,  transform=None, target_transform=None )

    # train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    # for i, (x, y) in enumerate(train_dataloader):
    #     print(i, x.shape, y.shape)
    #     break
# a = WeatherDataet('./Data')
# a.visualize(time='20080101-00', name='t2m')
# a.visualize(time='20080101-00', name='u10')