import numpy as np
import os
class Normalize():
    def __init__(self, data_dir, normalization_type = 'TAO', is_target = False):
        self.data_dir = data_dir
        self.normalization_type = normalization_type
        self.mean = None
        self.std = None
        self.is_target = is_target
        
    def __call__(self,x):
        if self.normalization_type == 'TAO':
            if self.mean is None:
                self.mean = np.load(os.path.join(self.data_dir, 'mean_TAO.npy'))
                self.std = np.load(os.path.join(self.data_dir, 'std_TAO.npy'))
            
            mean = self.mean.reshape((1,-1,1,1))
            std = self.std.reshape((1,-1,1,1))
            
            if self.is_target:
                mean = mean[:,-5:,:,:]
                std = std[:,-5:,:,:]
            return (x - mean) / std
        else:
            raise NotImplementedError
        