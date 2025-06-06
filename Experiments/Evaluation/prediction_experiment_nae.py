import os
import yaml
from pathlib import Path


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pdb


import lightning.pytorch as pl

import deepS2S.model.IndexLSTM as index_net
import deepS2S.model.ViTLSTM as vit_net

from deepS2S.utils.utils import statics_from_config
from deepS2S.utils.utils_data import load_data
from deepS2S.utils.utils_plot import *
from deepS2S.utils import utils_woo

# Set hyperparameters.

cm_list = ['#7fbf7b','#1b7837','#762a83','#9970ab','#c2a5cf']  
regimes = ['SB', 'NAO-', 'AR', 'NAO+'] #adapt if new regimes have been assigned

exd = os.path.dirname(os.path.abspath(__file__))
cfd = Path(exd).parent.absolute()

root_path = str(cfd.parent.absolute())+'/Data'
data_path = f"{root_path}/"
res_path = str(cfd.parent.absolute()) + f'/Data/Results/'


model_types = ['Index_LSTM', 'LSTM', 'ViT_LSTM']
for arch_type in model_types:
    if arch_type == 'ViT_LSTM':

            cfile = '_vit_lstm'
            config = yaml.load(open(f'{cfd}/config/config{cfile}.yaml'), Loader=yaml.FullLoader)


            strt_yr = config.get('strt','')
            trial_num = config.get('version', '')
            norm_opt = config.get('norm_opt','')
            arch = config.get('arch', 'ViT')
            tropics = config.get('tropics', '')

            stat_dir =  str(cfd.parent.absolute()) + f'/Data/Network/' + f'Statistics/{arch}'
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

            stat_dir =  str(cfd.parent.absolute()) + f'/Data/Network/' + f'Statistics/{arch_type}/'
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


            stat_dir =  str(cfd.parent.absolute()) + f'/Data/Network/' + f'Statistics/{arch_type}/'
            result_path = f'{res_path}Statistics/{arch_type}/'
            results_directory = Path(f'{result_path}version_{strt_yr}{trial_num}_{norm_opt}{tropics}/')
            os.makedirs(results_directory, exist_ok=True)
            mod_name = 'Index_LSTM'
            architecture = index_net.Index_LSTM

    config['net_root'] = str(cfd.parent.absolute()) + f'/Data/Network/'
    config['root'] = str(cfd.parent.absolute()) + f'/Data/Network/Sweeps/'
    config['data_root'] = root_path
    test_loader, data_set, cls_wt, test_set, infos = load_data(config)

    var_comb = config['var_comb']

    data_info, seasons = statics_from_config(config)

    # Load collected data.
    exp_dir =  f"{stat_dir}version_{strt_yr}{trial_num}_{norm_opt}{tropics}/"
    pths = [xs for xs in Path(exp_dir).iterdir() if xs.is_dir()]
    
    data_collect = np.load(f'{results_directory}/collected_loop_data_{len(pths)}.npz')
    data_result = np.load(f'{results_directory}/accuracy_{len(pths)}model.npz')
        
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
    loop_targets = np.repeat(targets[None,:,:], loop_classes.shape[0], axis = 0)

    quantile_step = 90
    lp_conf = loop_probabilities.flatten()
    qprob_90 = np.percentile(lp_conf,quantile_step)
    # Build loop baselines
    loop_tgs = loop_targets.reshape(loop_classes.shape[0]*loop_classes.shape[1],loop_classes.shape[2])
    loop_cls = loop_classes.reshape(loop_classes.shape[0]*loop_classes.shape[1],loop_classes.shape[2])
    loop_prbs = loop_probabilities.reshape(loop_classes.shape[0]*loop_classes.shape[1],loop_classes.shape[2],loop_probabilities.shape[3])
    nae_inputs = np.repeat(np.argmax(input_reg, axis = -1)[None,:,:], loop_classes.shape[0], axis = 0)


    # conditional probabilities.
    probability_nae, probability_nae_array, sampled_naes = utils_woo.nae_regimes_analysis_timesteps(nae_inputs, 
                         regimes, 
                         loop_probabilities, 
                         loop_targets,
                         qprob_90)


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

        lp_conf = loop_probabilities.flatten()

    q_all = 90
    qall_90 = np.percentile(lp_conf,q_all)
    count_mod_class_ts = np.zeros((len(regimes),loop_probabilities.shape[0],loop_probabilities.shape[2]))
    all_count =[] 
    for i in range(loop_probabilities.shape[0]): # num models
        count_mod = 0
        for j in range(loop_probabilities.shape[1]): # num samples
                for k in range(loop_probabilities.shape[2]):
                        if loop_probabilities[i,j,k,loop_targets[i,j,k]] > qall_90: 
                                count_mod_class_ts[loop_targets[i,j,k], i, k] +=1
                        count_mod +=1
        all_count.append(count_mod)
    all_count = np.array(all_count)

    np.savez(f'{results_directory}/{arch_type}nae_teleconnections.npz', probability_nae_array = probability_nae_array, vmax = vmax, 
        delta_t = delta_t, regimes = regimes, lead_weeks = lead_weeks, ylabs = ylabs, all_count = all_count, count_mod_class_ts = count_mod_class_ts, num_samples = loop_probabilities.shape[1])

    #np.savez(f'{results_directory}{arch_type}nae_teleconnections.npz', probability_nae_array = probability_nae_array, vmax = vmax, 
         #delta_t = delta_t, regimes = regimes, lead_weeks = lead_weeks, ylabs = ylabs, count_mod_class_ts = count_mod_class_ts,)