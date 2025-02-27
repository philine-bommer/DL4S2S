# functions to manipulate data before using it to train and test
# reshape, split into train and test set, create data loader
#
# functions to store data in files, generate plots (and store them)
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import torch
import pdb
import yaml


def move_tuple_to(t, device):
    tensors = []
    for tensor in t:
        tensors.append(tensor.to(device))
    return tuple(tensors)
  

def prod(iterable):
    prod = 1
    for elem in iterable:
        if elem is not None:
            prod *= elem
    return prod


def statics_from_config(config:  dict):

    var_comb = config['var_comb']
    data_info = {'vars': {},'config':{}}

    data_info['vars'][var_comb['input'][0]]={
                'dimension': '2d',
                'region': config['data'].get('region_olr','mjo'),
                'resolution': config['data']['resolution'],
                'type': 'continuous'}
        
    data_info['vars'][var_comb['input'][1]]= {
                'dimension': '2d',
                'region': config['data'].get('region_u','spv'),
                'resolution': config['data']['resolution'],
                'pressure_level': config['data']['pressure'],
                'type': 'continuous'}
    
    if 'index' in var_comb['input'][1]:
        data_info['vars'][var_comb['input'][1]]= {
                'dimension': '1d',
                'region': config['data'].get('region_u','spv'),
                'resolution': config['data']['resolution'],
                'pressure_level': config['data']['pressure'],
                'type': 'index'}
        
    if 'index' in var_comb['input'][0]:
        data_info['vars'][var_comb['input'][0]]=  {
                'dimension': '1d',
                'region': config['data'].get('region_olr','mjo'),
                'resolution': config['data']['resolution'],
                'type': 'categorical'}

    if 'Hindcast' in var_comb['input']:
        data_info['vars'][var_comb['input'][2]]= {
                'dimension': '1d',
                'type': 'categorical',
                'resolution': 2.5,
                'pressure_level': config['data']['nae'],
                'region': 'northern_hemi'
            }
        
        data_info['vars'][var_comb['output'][0]]= {
                    'dimension': '1d',
                    'type': 'categorical',
                    'resolution': 2.5,
                    'pressure_level': config['data']['nae'],
                    'region': 'northern_hemi'
                }
    else:
        data_info['vars'][var_comb['output'][0]]= {
                'dimension': '1d',
                'type': 'categorical',
                'resolution': config['data']['resolution'],
                'pressure_level': config['data']['nae'],
                'region': 'northern_hemi'
            }
    

    data_info[ 'config'] = {'regime_path':config['data'].get('regime_path',''),
            'data_path':config['data'].get('data_path',''),
            'strt':config.get('strt',''),
            'mean_days':config['data'].get('m_days',7),    
            'n_steps_in': config['data']['n_steps_in'],
            'n_steps_lag': config['data']['lag'],
            'n_steps_out': config['data']['n_steps_out'],
            'stack_maps': True,  # if False, each map is treated as a separate input. Not implemented yet.
            'test_set_name': config['data'][config['name']]['setname'],
            'test_set_seasons': list(range(config['data'][config['name']]['test_start'],config['data'][config['name']]['test_end']))
        }
        
        
    seasons =  {'train':{config['data']['dataset_name1']:list(range(config['data']['pre']['train_start'], config['data']['pre']['train_end'])),
                        config['data']['dataset_name2']:list(range(config['data']['fine']['train_start'], config['data']['fine']['train_end']))},
               'val':{config['data']['dataset_name1']:list(range(config['data']['pre']['val_start'], config['data']['pre']['val_end'])),
                      config['data']['dataset_name2']:list(range(config['data']['fine']['val_start'], config['data']['fine']['val_end']))},
               'test':{config['data']['dataset_name1']:list(range(config['data']['pre']['test_start'], config['data']['pre']['test_end'])),
                       config['data']['dataset_name2']:list(range(config['data']['fine']['test_start'], config['data']['fine']['test_end']))},}



    return data_info, seasons


def individual_timestep(data):

    outputs = []
    inputs = []
    for i in range(len(data)):

        ins, outs = data[i][0], data[i][1].numpy()
        input = ins[1].numpy()
        for j in range(len(outs)):
            outputs.append(outs[j])
            inputs.append(np.argmax(input, axis=1))

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    inputs = inputs.reshape(outputs.shape[0],int(input.shape[0]))

    return inputs, outputs

def concat_data(data1, data2):

    """_summary_

    Returns:
        _type_: _description_
    """

    X_train1, y_train1 = data1.train_data()
    X_val1, y_val1 = data1.val_data()
    X_test1, y_test1 = data1.test_data()

    X_train2, y_train2 = data2.train_data()
    X_val2, y_val2 = data2.val_data()
    X_test2, y_test2 =data2.test_data()

    x_train = np.concatenate((X_train1,X_train2))
    y_train = np.concatenate((y_train1,y_train2))
    x_val = np.concatenate((X_val1,X_val2))
    y_val = np.concatenate((y_val1,y_val2))
    # x_test = np.concatenate((X_test1,X_test2))
    # y_test = np.concatenate((y_test1,y_test2))

    return x_train, y_train, x_val, y_val, X_test2, y_test2

    # def align_predictions(x_test, y,y_prob):

    #     return pred, prob

def get_params_from_best_model(config:  dict,
                               architecure: str):
    """_summary_

    Args:
        config (dict): _description_
        architecure (str): architecure.__name__

    Returns:
        _type_: _description_
    """
    strt_yr = config.get('strt','')
    trial_num = config.get('version', '')
    norm_opt = config.get('norm_opt','')
    arch = config.get('arch','')
    name_var = config.get('tropics','')   
    exp_dir =  config['root'] + f'{arch}Sweep_{strt_yr}{trial_num}_{norm_opt}{name_var}/'
    
    try : 
        con_res = yaml.load(open(exp_dir + '/result.yml'), Loader=yaml.UnsafeLoader)
    except: 
        exp_dir = config['root']
        con_res = yaml.load(open(exp_dir + 'result.yml'), Loader=yaml.UnsafeLoader)

    hp_dir = con_res.get("test_dir",0)
    hps = yaml.load(open(hp_dir / 'hparams.yaml'), Loader=yaml.UnsafeLoader)
    conv_params = {}
  
    if 'Conv' in architecure:
        conv_params['decoder_hidden_dim'] = hps['decoder_hidden_dim']
        conv_params['learning_rate'] = hps['learning_rate']
        conv_params['dropout'] = hps['dropout']
        conv_params['weight_decay'] = hps['weight_decay']
        conv_params['lrs'] = hps['lr_fine']
        conv_params['encoder_num_layers'] = hps['encoder_num_layers']
        conv_params['hidden_dim'] = hps['hidden_dim']
        conv_params['norm_both'] = hps['norm_both']
        conv_params['norm_bch'] = hps['norm_bch']
        conv_params['norm'] = hps['norm']
        conv_params['output_probabilities'] = hps['output_probabilities']
        conv_params['swa'] = hps['swa']
        conv_params['bs'] = hps['bs']
        conv_params['gamma'] = hps['gamma']
        conv_params['gc_fine'] = hps['grad_clip_fine']
        conv_params['gc_pre'] = hps['grad_clip_pre']

    elif 'spatiotemporal' in architecure:

        conv_params['decoder_hidden_dim'] = hps['decoder_hidden_dim']
        conv_params['learning_rate'] = hps['learning_rate']
        conv_params['dropout'] = hps['dropout']
        conv_params['weight_decay'] = hps['weight_decay']
        if not "fine" in config['setting_training']:
            conv_params['lrs'] = hps['lr_fine']
        gamma = {}
        gamma['val'] = 3
        gamma['nurmeric'] = True
        conv_params['gc_fine'] = hps['grad_clip_fine']
        conv_params['gc_pre'] = hps['grad_clip_pre']

        # conv_params['hidden_dim'] = hps['hidden_dim']
        conv_params['norm_both'] = hps['norm_both']
        conv_params['norm_bch'] = hps['norm_bch']
        conv_params['norm'] = hps['norm']
        conv_params['output_probabilities'] = hps['output_probabilities']
        conv_params['swa'] = hps['swa']
        conv_params['bs'] = hps['bs']
        conv_params['gamma'] = gamma
    else: 
        conv_params['hidden'] = hps['decoder_hidden_dim']
        conv_params['learning_rate'] = hps['learning_rate']
        conv_params['weight_decay'] = hps['weight_decay']
        conv_params['lrs'] = hps['lr_fine']
        conv_params['bs'] = hps['bs']
        conv_params['swa'] = hps['swa']
        conv_params['gamma'] = hps['gamma']
        conv_params['gc_fine'] = hps['grad_clip_fine']
        conv_params['gc_pre'] = hps['grad_clip_pre']

    return conv_params

def get_params_from_model_obj(config:  dict,
                               architecture: any):
    """_summary_

    Args:
        config (dict): _description_
        architecure (str): architecure.__name__

    Returns:
        _type_: _description_
    """
    
    strt_yr = config.get('strt','')
    trial_num = config.get('version', '')
    norm_opt = config.get('norm_opt','')
    name_var = config.get('tropics','')   
    exp_dir =  config['root'] + f'{arch}Sweep_{strt_yr}{trial_num}_{norm_opt}{name_var}/'
  
    try : 
        con_res = yaml.load(open(exp_dir + '/result.yml'), Loader=yaml.UnsafeLoader)
    except: 
        exp_dir = config['root']
        con_res = yaml.load(open(exp_dir + 'result.yml'), Loader=yaml.UnsafeLoader)

    hp_dir = con_res.get("test_dir",0)
    hps = yaml.load(open(hp_dir / 'hparams.yaml'), Loader=yaml.UnsafeLoader)
    model = architecture.load_from_checkpoint(f"{hp_dir}/best_finetuned_model.ckpt", 
                                                      cls_wgt = hps['cls_wgt'], 
                                                      strict=False)

    if config.get('saved', True):
        if 'Conv' in architecture.__name__:
            conv_params = model.hparams
            conv_params['lrs'] = hps['lr_fine']
            conv_params['swa'] = hps['swa']
            conv_params['bs'] = hps['bs']
        else: 
            conv_params = {}
            conv_params['hidden'] = model.hparams['decoder_hidden_dim']
            conv_params['learning_rate'] = model.hparams['learning_rate']
            conv_params['weight_decay'] = model.hparams['weight_decay']
            conv_params['lrs'] = hps['lr_fine']
            conv_params['bs'] = hps['bs']
            conv_params['swa'] = hps['swa']
            conv_params['gc_fine'] = hps['grad_clip_fine']
            conv_params['gc_pre'] = hps['grad_clip_pre']
            if not "fine" in config['setting_training']:
                conv_params['lrs'] = hps['lr_fine']

    else:
        if 'Conv' in architecture.__name__:
            conv_params = {}
            conv_params['norm_both'] = config['network']['norm_both']
            conv_params['weight_decay'] = config['network']['weight_decay']
            conv_params['decoder_hidden_dim'] = config['network']['decoder_hidden_dim']
            conv_params['dropout'] = config['network']['dropout']
            conv_params['learning_rate'] = config['network']['learning_rate']
            conv_params['norm'] = config['network']['norm']
            conv_params['norm_bch'] = config['network']['norm_bch']
            conv_params['lrs'] = config['network']['lrs']
            conv_params['swa'] = config['network']['swa']
            conv_params['bs'] = config['data']['bs']

        elif 'spatiotemporal' in architecture.__name__:
            conv_params = {}
            conv_params['norm_both'] = config['network']['norm_both']
            conv_params['weight_decay'] = config['network']['weight_decay']
            conv_params['hidden_dim'] = config['network']['hidden_dim']
            conv_params['encoder_num_layers'] = config['network']['encoder_num_layers']
            conv_params['decoder_hidden_dim'] = config['network']['decoder_hidden_dim']
            conv_params['dropout'] = config['network']['dropout']
            conv_params['learning_rate'] = config['network']['learning_rate']
            conv_params['norm'] = config['network']['norm']
            conv_params['norm_bch'] = config['network']['norm_bch']
            conv_params['lrs'] = config['network']['lrs']
            conv_params['swa'] = config['network']['swa']
            conv_params['bs'] = config['data']['bs']
            conv_params['gc_fine'] = hps['grad_clip_fine']
            conv_params['gc_pre'] = hps['grad_clip_pre']

        else: 
            conv_params = {}
            conv_params['hidden'] = config['network']['decoder_hidden_dim']
            conv_params['learning_rate'] = config['network']['learning_rate']
            conv_params['weight_decay'] = config['network']['weight_decay']
            conv_params['lrs'] = config['network']['lrs']
            conv_params['bs'] = config['network']['bs']
            conv_params['swa'] = config['network']['swa']
    

    return conv_params

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


def get_random_seasons(end_season: int, ratios: list):

    # Generate random indices
    random_indices = np.random.permutation(end_season)

    # Define the split ratio
    train_ratio = ratios[0]
    train_size = int(train_ratio * end_season)
    val_size = int(ratios[1] * train_size)

    # Split the indices into training and testing sets
    train_val_indices = list(random_indices[:train_size])
    train_indices = list(train_val_indices[val_size:train_size])
    val_indices = list(train_val_indices[:val_size])
    test_indices = list(random_indices[train_size:])

    print("Train Seasons:", train_indices)
    print("Val Seasons:", val_indices)
    print("Test Seasons:", test_indices)

    return train_indices, val_indices, test_indices


