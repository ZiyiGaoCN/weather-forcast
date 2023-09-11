"""
Code to Evaluate of using the Mean Absolute Scaled Error (MASE) and the Symmetric Mean Absolute Percentage Error (SMAPE)
of ForecastNet for a given test set.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import numpy as np
import torch
from dataHelpers import format_input
from calculateError import calculate_error

def evaluate(fcstnet, test_dataloader, return_lists=False):
    """
    Calculate various error metrics on a test dataset
    :param fcstnet: A forecastNet object defined by the class in forecastNet.py
    :param test_x: Input test data in the form [encoder_seq_length, n_batches, input_dim]
    :param test_y: target data in the form [encoder_seq_length, n_batches, input_dim]
    :return: mase: Mean absolute scaled error
    :return: smape: Symmetric absolute percentage error
    :return: nrmse: Normalised root mean squared error
    """
    fcstnet.model.eval()

    # # Load model parameters
    # checkpoint = torch.load(fcstnet.save_file, map_location=fcstnet.device)
    # fcstnet.model.load_state_dict(checkpoint['model_state_dict'])
    # fcstnet.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    with torch.no_grad():
        # Format the inputs
        # Inference
        y_pred_list = []
        # Compute outputs for a mixture density network output
        if fcstnet.model_type == 'dense' or fcstnet.model_type == 'conv':
            n_forecasts = 20
            for i in range(n_forecasts):
                y_pred, mu, sigma = fcstnet.model(test_x, test_y, is_training=False)
                y_pred_list.append(y_pred)
            y_pred = torch.mean(torch.stack(y_pred_list), dim=0)
        # Compute outputs for a linear output
        elif fcstnet.model_type == 'dense2' or fcstnet.model_type == 'conv2':
            y_pred = fcstnet.model(test_x, test_y, is_training=False)
        elif fcstnet.model_type == 'UNet':
            for idx, (input) in enumerate(test_dataloader):
                input = input[0].to(fcstnet.device)
                B, in_seq, C_in, H, W = input.shape
                input = input.view(B, in_seq*C_in, H, W)
                outputs = fcstnet.model(input)
                outputs_formatted = outputs.view(B, 20, 5, H, W)
                y_pred_list.append(outputs_formatted.cpu().numpy())
                
            y_pred = np.concatenate(y_pred_list, axis=0)
        return y_pred

if __name__ == '__main__':
    pass
    