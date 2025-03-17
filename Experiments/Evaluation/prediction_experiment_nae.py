import os
import yaml
from pathlib import Path


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


import lightning.pytorch as pl

import deepS2S.model.IndexLSTM as index_net
import deepS2S.model.ViTLSTM as vit_net

from deepS2S.utils.utils import statics_from_config
from deepS2S.utils.utils_data import load_data
from deepS2S.utils.plot_utils import *
import deepS2S.utils.utils_evaluation as eval
import deepS2S.utils.utils_woo as woo

# Set hyperparameters.

cm_list = ['#7fbf7b','#1b7837','#762a83','#9970ab','#c2a5cf']  #762a83
regimes = ['SB', 'NAO-', 'AR', 'NAO+']

exd = os.path.dirname(os.path.abspath(__file__))
cfd = exd.parent.absolute()

root_path = str(cfd.parent.absolute())+'/Data'
data_path = f"{root_path}/"
res_path = str(cfd.parent.absolute()) + f'/Data/Results/'


arch_types = ['Index_LSTM', 'LSTM', 'ViT']
for arch_type in arch_types:
    if arch_type == 'ViT':

            cfile = '_vit_lstm'
            config = yaml.load(open(f'{cfd}/config/config{cfile}.yaml'), Loader=yaml.FullLoader)


            strt_yr = config.get('strt','')
            trial_num = config.get('version', '')
            norm_opt = config.get('norm_opt','')
            arch = config.get('arch', 'ViT')
            tropics = config.get('tropics', '')

            stat_dir =  config['net_root'] + f'Statistics/{arch}'
            result_path = f'{res_path}/Statistics/{arch}/'
            results_directory = Path(f'{result_path}version_{strt_yr}{trial_num}_{norm_opt}{tropics}/')
            os.makedirs(results_directory, exist_ok=True)


            mod_name = 'ViT_LSTM'
            architecture = vit_net.ViT_LSTM
            
    elif arch_type == 'LSTM':
            cfile = '_lstm'
            config = yaml.load(open(f'{cfd}/config/config{cfile}.yaml'), Loader=yaml.FullLoader)


            strt_yr = config.get('strt','')
            trial_num = config.get('version', '')
            norm_opt = config.get('norm_opt','')
            arch = config.get('arch', 'ViT')
            tropics = config.get('tropics', '')

            stat_dir =  config['net_root'] + f'Statistics/{arch_type}/'
            result_path = f'{res_path}Statistics/{arch_type}/'
            results_directory = Path(f'{result_path}version_{strt_yr}{trial_num}_{norm_opt}{tropics}/')
            os.makedirs(results_directory, exist_ok=True)

            mod_name = 'ViT_LSTM'
            architecture = vit_net.ViT_LSTM

    else:
            config = yaml.load(open(f'{cfd}/config/config_index_lstm.yaml'), Loader=yaml.FullLoader)

            config_base = yaml.load(open(f'{cfd}/config/config_vit_lstm.yaml'), Loader=yaml.FullLoader)

            strt_yr = config.get('strt','')
            trial_num = config.get('version', '')
            norm_opt = config.get('norm_opt','')
            arch = config.get('arch', 'ViT')
            tropics = config.get('tropics', '')


            stat_dir =  config['net_root'] + f'Statistics/{arch_type}/'
            result_path = f'{res_path}Statistics/{arch_type}/'
            results_directory = Path(f'{result_path}version_{strt_yr}{trial_num}_{norm_opt}{tropics}/')
            os.makedirs(results_directory, exist_ok=True)
            mod_name = 'Index_LSTM'
            architecture = index_net.IndexLSTM

    test_loader, data_set, cls_wt, test_set, infos = load_data(config)

    var_comb = config['var_comb']

    data_info, seasons = statics_from_config(config)

    # Load collected data.
    exp_dir =  f"{stat_dir}version_{strt_yr}{trial_num}_{norm_opt}{tropics}/"
    pths = [xs for xs in Path(exp_dir).iterdir() if xs.is_dir()]

    data_collect = np.load(f'{results_directory}/collected_loop_data_{len(pths)-1}.npz')
    data_result = np.load(f'{results_directory}/accuracy_{len(pths)-1}model.npz')
        
    persistance = data_collect['persistance'] 
    dates = data_collect['dates'] 
    daytimes = data_collect['daytimes']
    loop_probabilities = data_collect['loop_probabilities']
    loop_classes = data_collect['loop_classes']
    predictions_baseline = data_collect['predictions_baseline']
    targets = data_collect['targets']

    input_reg = []
    for input, output, weeks, days in data_set:
        if arch_type == 'Index_LSTM':
            input_reg.append(input[1][None,:,-4:].numpy().squeeze())
        else:
            input_reg.append(np.array(input[1]).squeeze())


    input_reg = np.concatenate(input_reg).reshape((predictions_baseline.shape[0],
                                                            predictions_baseline.shape[1],4))
    
    # Build loop baselines
    loop_targets = np.repeat(targets[None,:,:], loop_classes.shape[0], axis = 0)
    lp_conf = loop_probabilities.flatten()
    q_all, q_90 = 85, 90
    qall_90 = np.percentile(lp_conf,q_all)
    loop_tgs = loop_targets.reshape(loop_classes.shape[0]*loop_classes.shape[1],loop_classes.shape[2])
    loop_cls = loop_classes.reshape(loop_classes.shape[0]*loop_classes.shape[1],loop_classes.shape[2])
    loop_prbs = loop_probabilities.reshape(loop_classes.shape[0]*loop_classes.shape[1],loop_classes.shape[2],loop_probabilities.shape[3])
    nae_inputs = np.repeat(np.argmax(input_reg, axis = -1)[None,:,:], loop_classes.shape[0], axis = 0)


    # conditional probabilities.
    probability_nae, probability_nae_array, sampled_naes = woo.nae_regimes_analysis_timesteps(nae_inputs, 
                         regimes, 
                         loop_probabilities, 
                         loop_targets)


    t_in, t_out = nae_inputs.shape[2],loop_probabilities.shape[2]
    vmin = np.abs(np.nanmin(probability_nae_array))
    vmax = np.nanmax(probability_nae_array)

    vmax = 55
    delta_t = [-(t+1) for t in range(t_in +t_out-1)]
    lead_weeks = [f"t+{k+1}" for k in range(nae_inputs.shape[2])]
    ylabs = []
    for i in range(len(regimes)):
        for j in range(len(lead_weeks)):
            ylabs.append(lead_weeks[j])

    np.savez(f'{results_directory}{arch_type}nae_teleconnections.npz', probability_nae_array = probability_nae_array, vmax = vmax, 
         delta_t = delta_t, regimes = regimes, lead_weeks = lead_weeks, ylabs = ylabs)