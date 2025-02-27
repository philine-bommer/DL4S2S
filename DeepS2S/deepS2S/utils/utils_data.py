import os
import json
import shutil
from datetime import datetime
import pdb

import torch
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from utils import statics_from_config
from dataset.datasets_wrapped import TransferData, WeatherDataset

def _get_cv_data_split(full_data, i):
    k_fold = full_data['config']['k_fold']

    n_samples = len(full_data['train'])
    indicies = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indicies)
    indicies = np.tile(indicies, 2)
    shift = n_valid = n_samples // k_fold
    n_train = n_samples - n_valid

    init = i*shift
    train_indicies = indicies[init:init+n_train]
    valid_indicies = indicies[init+n_train:init+n_train+n_valid]

    data = full_data.copy()
    data['train'] = torch.utils.data.dataset.Subset(full_data['train'], train_indicies)
    data['valid'] = torch.utils.data.dataset.Subset(full_data['train'], valid_indicies)

    return data

def cls_weights(data,
                ctype = 'balanced'):
    
    outputs = []
    for i in range(data['train'].__len__()):
        outputs.append(data['train'][i][1].numpy())
    otps = np.array(outputs)
    
    cls_wt = class_weight.compute_class_weight(ctype, 
                            classes = np.unique(otps), 
                            y = otps.flatten())

    return cls_wt


def load_data(config):
    # pl.seed_everything(42)

    var_comb = config['var_comb']

    data_info, seasons = statics_from_config(config)

    infos = [data_info, var_comb, seasons]
        
    # Create data loader.
    params = {'seasons': seasons, 'test_set_name':config['data'][config['name']]['setname']}

    Fine_data = TransferData(config['data']['dataset_name2'], 
                                var_comb, data_info, config['data']['bs'], **params)
    Fine_data.train_dataloader()
    Fine_data.val_dataloader()
    test_loader = Fine_data.test_dataloader()
    data = Fine_data.access_dataset()

    cls_wt = cls_weights(data, 'balanced')
   
    
    data_set = WeatherDataset(
        dataset_name=config['data']['dataset_name2'],
        data_info=data_info,
        var_comb=var_comb,
        seasons=seasons['test'][config['data']['dataset_name2']],
        return_dates=True
    )

    test_set = WeatherDataset(
        dataset_name=config['data']['dataset_name2'],
        data_info=data_info,
        var_comb=var_comb,
        seasons=seasons['test'][config['data']['dataset_name2']]
    )
    
    return test_loader, data_set, cls_wt, test_set, infos

def extract_data_arrays(data_set, pred, pred_baseline):
    dates = []
    daytimes = []
    targets = []
    persistance = []
    i = 0

    for input, output, weeks, days in data_set:
        targets.append(np.array(output).squeeze())
        dates.append(np.array(weeks).squeeze())
        daytimes.append(np.array(days).squeeze())

        persistance.append(np.repeat(np.argmax(input[1].numpy()[-1]), 6).T)
        i += 1
        
    predictions = []
    predictions_baseline = []
    for i in range(len(pred)):
        predictions.append(pred[i])
        predictions_baseline.append(pred_baseline[i])
    pdb.set_trace()
    for var in ['predictions', 'predictions_baseline']:
        locals()[var] = np.concatenate(locals()[var])
    for var in ['dates', 'daytimes', 'targets', 'persistance']:
        locals()[var] = np.concatenate(locals()[var]).reshape((predictions.shape[0],predictions.shape[1]))
    
    return dates, daytimes, targets, persistance, predictions, predictions_baseline

def generate_clim_pred(clim, dates):

    if clim.ndim == 2:
        samples, n_classes = clim.shape
        samples, n_step = dates.shape
        clim_pred = np.zeros((*dates.shape,n_classes))
    else:
        samples, n_step = dates.shape
        clim_pred = np.zeros(dates.shape)
    
    for i in range(n_step):
        for j in range(samples):
            if clim.ndim == 2:
                clim_pred[j,i,:] = clim.sel(dayoftheyear=dates[j,i]).values
            else:
                clim_pred[j,i] = clim.sel(dayofyear=dates[j,i]).values
    
    if not clim.ndim == 2:
            clim_pred = clim_pred.astype(int)
        
    return clim_pred



