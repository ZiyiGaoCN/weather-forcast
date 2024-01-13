import pandas
from omegaconf import OmegaConf
import argparse
import yaml
import numpy as np
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))

from dataset.dataset_torch import WeatherDataet_numpy
import torch
from weather_forcast.validate import validate_onestep
from weather_forcast.utils import initial_model
from loguru import logger 

parser = argparse.ArgumentParser(description="hello")
parser.add_argument('--cfg_path', default=None, type=str, help='node rank for distributed training')
parser.add_argument('--ckpt_path', default=None, type=str, help='node rank for distributed training')
parser.add_argument('--device', default='cuda:7', type=str, help='node rank for distributed training')
args = parser.parse_args()

with open(args.cfg_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

# Convert to omegaconf.DictConfig
config = OmegaConf.create(yaml_data)
config_deepspeed = OmegaConf.create({"deepspeed":vars(args)})
config = OmegaConf.merge(config, config_deepspeed)
cfg = config




base_path = cfg.data.npy_name
data_ori = np.memmap(base_path, dtype = 'float32',mode = 'c', shape = tuple(cfg.data.shape) , order = 'C')
    
train_set = WeatherDataet_numpy(data_ori,range=[55000,55480],step=1)
valid_set = WeatherDataet_numpy(data_ori,range=[56000,57000],step=1)
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_set, batch_size=160, shuffle=True, num_workers=4)
dataloader = DataLoader(valid_set, batch_size=160, shuffle=True, num_workers=4)

ckpt = torch.load(args.ckpt_path,map_location='cpu')
state_dict = ckpt['module']
# print(ckpt.keys())
if 'model_params':
    model_params = ckpt['model_params']
    logger.info('load model params from ckpt')
else:
    model_params = cfg.model
    logger.info('load model params from config')
    # model_params.param.time_embed=False
# print(model_params)
# print(model_params)

name = args.ckpt_path.split('/')[-3]
iter_num = args.ckpt_path.split('/')[-2]

concat_name = name + '_' + iter_num

try:
    os.mkdir(f'evaluation/{concat_name}')
except:
    logger.warning('dir exists')

if not hasattr(model_params.param,'uncertainty_loss'):
    model_params.param.uncertainty_loss = False

model = initial_model(model_params)
model.load_state_dict(state_dict) 
model.to(args.device)

logger.info('start eval on train set')

train_rmses,train_accs = validate_onestep(model,train_dataloader)
train_rmses = np.array(train_rmses)
train_accs = np.array(train_accs)

np.save(f'evaluation/{concat_name}/trainone_rmses.npy',train_rmses)
np.save(f'evaluation/{concat_name}/trainone_accs.npy',train_accs)

logger.info('start eval on valid set')

rmses,accs = validate_onestep(model,dataloader)
rmses = np.array(rmses)
accs = np.array(accs)

np.save(f'evaluation/{concat_name}/validone_rmses.npy',rmses)
np.save(f'evaluation/{concat_name}/validone_accs.npy',accs)
