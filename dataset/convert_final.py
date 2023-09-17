from dataset import  load_dataset
import numpy as np
import os
import sys

save_dir = '../../npy_data'
raw_data_dir = '../../Data'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

total_data = load_dataset(raw_data_dir, 2007, 2012).x

times = total_data.time.values
channels = total_data.channel.values
lats = total_data.lat.values
lons = total_data.lon.values


np.save(os.path.join(save_dir, 'times'), times)
np.save(os.path.join(save_dir, 'channels'), channels)
np.save(os.path.join(save_dir, 'lats'), lats)
np.save(os.path.join(save_dir, 'lons'), lons)


data = np.memmap(os.path.join(save_dir, 'data.npy'), dtype = 'float32',mode = 'w+', shape = (7304, 70, 161, 161) , order = 'C')

time_acc = 0
for q in range(2007, 2012):
    a = load_dataset(raw_data_dir, q, q+ 1).x
    data[time_acc: time_acc + len(a.time.values), :, :, :] = a.values
    time_acc += len(a.time.values)

data.flush()

print(1)
mean = data.mean(axis = 0, keepdims = True)
print(2)
std = data.std(axis = 0, keepdims = True)
print(3)

data_normalize = np.memmap(os.path.join(save_dir, 'data_normailze.npy'), dtype = 'float32',mode = 'w+', shape = (7304, 70, 161, 161) , order = 'C')
print(4)
data_normailze = (data -mean) / std
print(5)
data_normailze.flush()
print(6)


