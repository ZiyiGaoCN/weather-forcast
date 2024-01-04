import numpy as np
import torch
import xarray as xr
import importlib
import pandas as pd

constants = xr.open_dataset('/home/gaoziyi/weather/weather_bench/bench/constants/constants_5.625deg.nc')
stds = pd.read_csv('/home/gaoziyi/weather/weather_bench/concatenated_array_mean_std.csv')['std'].values
# means = np.load('/home/gaoziyi/weather/weather_bench/concatenated_array_mean.npy')
# stds = np.load('/home/gaoziyi/weather/weather_bench/concatenated_array_std.npy')

def get_loss_weights():
    weights = np.linspace(0.005,0.065,13)
    # 5 times weights
    weights = weights.repeat(5)
    weights = np.concatenate([weights,np.array([0.1,0.1,1.0])],axis=0)
    return weights

def log_indices():
    return {
        '2m': -1,
        '10v': -2,
        '10u': -3,
        'z500': 45,
        't850': 10,
    }

def get_stds():
    return stds

def calculate_rmse(actual, predicted, mean_dims=(2,3)):
    # [B, C, H=32, W=64]
    if isinstance(actual, torch.Tensor):
        act = actual.detach().cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        pred = predicted.detach().cpu().numpy()
    error = act - pred
    latitude = constants.lat.values
    weights_lat = np.cos(np.deg2rad(latitude))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(( (error) ** 2 * weights_lat[None,None,:,None]).mean(axis=mean_dims))
    return rmse*stds
    

def calculate_acc(actual, predicted, mean_dims=(2,3)):
    # [B, C, H=32, W=64]
    if isinstance(actual, torch.Tensor):
        act = actual.detach().cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        pred = predicted.detach().cpu().numpy()
    
    latitude = constants.lat.values
    weights_lat = np.cos(np.deg2rad(latitude))
    weights_lat /= weights_lat.mean()

    relate = weights_lat[None,None,:,None] * act * pred
    relate = relate.sum(axis=(2,3))
    norm_A = (weights_lat[None,None,:,None] * (act**2))
    norm_B = (weights_lat[None,None,:,None] * (pred**2))
    norm_A = norm_A.sum(axis=(2,3))
    norm_B = norm_B.sum(axis=(2,3))
    norm = np.sqrt(norm_A * norm_B)
    acc = relate / norm
    return acc


def initial_model(param):
    model_path = param.model_path
    model_type = param.model_type
    model_module = importlib.import_module(model_path)
    model_class = getattr(model_module, model_type)
    model = model_class(**param.param)
    return model