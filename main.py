"""
This script demonstrates initialisation, training, evaluation, and forecasting of ForecastNet. The dataset used for the
time-invariance test in section 6.1 of the ForecastNet paper is used for this demonstration.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import numpy as np
import matplotlib.pyplot as plt
from forecastNet import forecastNet
from train import train
from evaluate import evaluate
from dataHelpers import generate_data
from dataset.loading import loading
import wandb

from dataset.dataset_mem import split_dataset
from torch.utils.data import DataLoader

from dataset.transform.transform import Normalize, InverseNormalize

#Use a fixed seed for repreducible results
np.random.seed(1)



# Model parameters
model_type = 'Swin-UNet' #'dense' or 'conv', 'dense2' or 'conv2'
in_seq_length = 2
out_seq_length = 20
hidden_dim = 96
input_dim = 70
output_dim = 5
learning_rate = 0.0001
n_epochs= 100
batch_size = 16
weight_decay= 1e-5

wandb.init(project="weather-forecast",name="Swin_UNet_run")

# Initialise model
fcstnet = forecastNet(in_seq_length=in_seq_length, out_seq_length=out_seq_length, input_dim=input_dim,
                        hidden_dim=hidden_dim, output_dim=output_dim, model_type = model_type, batch_size = batch_size,
                        n_epochs = n_epochs, learning_rate = learning_rate, weight_decay=1e-5, 
                        save_file = './forecastnet.pt',device = "cuda:1")

transform_train = Normalize('../dataset/normalization', normalization_type = 'TAO', is_target = False)
transform_target = Normalize('../dataset/normalization',  normalization_type = 'TAO', is_target = True)
inverse_transform_target = InverseNormalize('../dataset/normalization',  normalization_type = 'TAO', is_target = True)

train_data, valid_data = split_dataset('../dataset',ratio=0.8,
                                       transform=None,
                                       target_transform=None)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True,num_workers=96)
valid_dataloader = DataLoader(valid_data, batch_size=64, shuffle=False,num_workers=24)

# Train the model
training_costs, validation_costs = train(fcstnet, train_dataloader,valid_dataloader, restore_session=False, wandb=wandb, inverse_transform_target = None)
# Plot the training curves
plt.figure()
plt.plot(training_costs)
plt.plot(validation_costs)

# Evaluate the model
mase, smape, nrmse = evaluate(fcstnet, test_x, test_y, return_lists=False)
print('')
print('MASE:', mase)
print('SMAPE:', smape)
print('NRMSE:', nrmse)

# Generate and plot forecasts for various samples from the test dataset
samples = [0, 500, 1039]
# Models with a Gaussian Mixture Density Component output
if model_type == 'dense' or model_type == 'conv':
    # Generate a set of n_samples forecasts (Monte Carlo Forecasts)
    num_forecasts = 10
    y_pred = np.zeros((test_y.shape[0], len(samples), test_y.shape[2], num_forecasts))
    mu = np.zeros((test_y.shape[0], len(samples), test_y.shape[2], num_forecasts))
    sigma = np.zeros((test_y.shape[0], len(samples), test_y.shape[2], num_forecasts))
    for i in range(num_forecasts):
        y_pred[:, :, :, i], mu[:, :, :, i], sigma[:, :, :, i] = fcstnet.forecast(test_x[:, samples, :])
    s_mean = np.mean(y_pred, axis=3)
    s_std = np.std(y_pred, axis=3)
    botVarLine = s_mean - s_std
    topVarLine = s_mean + s_std

    for i in range(len(samples)):
        plt.figure()
        plt.plot(np.arange(0, in_seq_length), test_x[:, samples[i], 0],
                 '-o', label='input')
        plt.plot(np.arange(in_seq_length, in_seq_length + out_seq_length), test_y[:, samples[i], 0],
                 '-o', label='data')
        plt.plot(np.arange(in_seq_length, in_seq_length + out_seq_length), s_mean[:, i, 0],
                 '-*', label='forecast')
        plt.fill_between(np.arange(in_seq_length, in_seq_length + out_seq_length),
                         botVarLine[:, i, 0], topVarLine[:, i, 0],
                         color='gray', alpha=0.3, label='Uncertainty')
        plt.legend()
# Models with a linear output
elif model_type == 'dense2' or model_type == 'conv2':
    # Generate a forecast
    y_pred = fcstnet.forecast(test_x[:,samples,:])

    for i in range(len(samples)):
        # Plot the forecast
        plt.figure()
        plt.plot(np.arange(0, fcstnet.in_seq_length),
                 test_x[:, samples[i], 0],
                 'o-', label='test_data')
        plt.plot(np.arange(fcstnet.in_seq_length, fcstnet.in_seq_length + fcstnet.out_seq_length),
                 test_y[:, samples[i], 0],
                 'o-')
        plt.plot(np.arange(fcstnet.in_seq_length, fcstnet.in_seq_length + fcstnet.out_seq_length),
                 y_pred[:, i, 0],
                 '*-', linewidth=0.7, label='mean')

plt.show()
