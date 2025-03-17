import os
import yaml
from pathlib import Path


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import lightning.pytorch as pl

import deepS2S.model.IndexLSTM as index_net
import deepS2S.model.ViTLSTM as vit_net

from deepS2S.utils.utils import statics_from_config
from deepS2S.utils.utils_data import generate_clim_pred, load_data
from deepS2S.utils.plot_utils import *
import deepS2S.utils.utils_evaluation as eval

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
    olr = data_collect['olr'] 
    u10 = data_collect['u10'] 
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

    q_90 = 90
    # Build loop baselines
    loop_u10 = np.repeat(u10[None,...], loop_classes.shape[0],axis=0)
    loop_olr = np.repeat(olr[None,...], loop_classes.shape[0],axis=0)
    loop_tgs = loop_targets.reshape(loop_classes.shape[0]*loop_classes.shape[1],loop_classes.shape[2])
    loop_cls = loop_classes.reshape(loop_classes.shape[0]*loop_classes.shape[1],loop_classes.shape[2])
    loop_prbs = loop_probabilities.reshape(loop_classes.shape[0]*loop_classes.shape[1],loop_classes.shape[2],loop_probabilities.shape[3])


    index_path = f'{data_path}Index'
    mjo_index = np.load(f'{index_path}/MJO_index_1980-2023_mjo_testset.npz')
    pv_index = np.load(f'{index_path}/pv_index_1980-2023_spv_testset.npz')

    mjo_index_in = mjo_index['input']
    mjo_index_out = mjo_index['output']
    daytimes = mjo_index['daytimes']
    dates = mjo_index['dates']

    pv_index_in = pv_index['input']
    pv_index_out = pv_index['output']

    pv_inputs = np.repeat(pv_index_in[None,:,:], loop_classes.shape[0], axis = 0)
    pv_outputs = np.repeat(pv_index_out[None,:,:], loop_classes.shape[0], axis = 0)
    mjo_inputs = np.repeat(mjo_index_in[None,:,:], loop_classes.shape[0], axis = 0)
    mjo_outputs = np.repeat(mjo_index_out[None,:,:], loop_classes.shape[0], axis = 0)

    lp_conf = loop_probabilities.flatten()
    qall_90 = np.percentile(lp_conf,q_90)
    pred_all_90 = []
    ts_all_90 = []

    for i in range(loop_probabilities.shape[0]):
        for j in range(loop_probabilities.shape[1]):
            for k in range(loop_probabilities.shape[2]):
                for l in range(loop_probabilities.shape[3]):
                    if loop_probabilities[i,j,k,l] > qall_90: 
                        ts_all_90.append(f'lead week {k+1}')
                        pred_all_90.append(l)

    pred_all_90 = np.array(pred_all_90)

    lp_conf = loop_probabilities.flatten()
    q_all = 90
    qall_90 = np.percentile(lp_conf,q_all)
    sum_spv_reg = np.zeros((4,1))
    cnts_reg_in  = np.zeros((4,1))


    for i in range(loop_probabilities.shape[0]):
        for j in range(loop_probabilities.shape[1]):
            for k in range(loop_probabilities.shape[2]):
                for t in range(pv_inputs.shape[2]):
                    if not np.isnan(pv_inputs[i,j,t]):
                        cnts_reg_in[input_reg[j,k]] += 1
                        sum_spv_reg[input_reg[j,k]] += pv_inputs[i,j,t]

    avg_spv = (sum_spv_reg.T/cnts_reg_in).flatten()


    pv_anomalies = np.zeros(pv_inputs.shape)

    for i in range(loop_probabilities.shape[0]):
        for j in range(loop_probabilities.shape[1]):
            for k in range(loop_probabilities.shape[2]):
                for t in range(pv_inputs.shape[2]):
                        if not np.isnan(pv_inputs[i,j,t]):
                            pv_anomalies[i,j,t] = pv_inputs[i,j,t]- avg_spv[input_reg[j,k]]

    t_in, t_out = pv_inputs.shape[2],loop_probabilities.shape[2]


    pv_anom_reg_t = []
    pv_inp_reg_t = []
    q90_labs = [] 
    for i in range(loop_probabilities.shape[0]): # number models
        pv_reg_t = np.zeros((4,t_in, t_out))
        pv_inp_t = np.zeros((4,t_in, t_out))
        cnt_correct_t = np.zeros((4,t_in, t_out))
        for j in range(loop_probabilities.shape[1]): # number weeks/ samples
            for k in range(loop_probabilities.shape[2]): # output timesteps
                for t in range(pv_inputs.shape[2]): #input timesteps
                    if not np.isnan(pv_inputs[i,j,t]):
                        if loop_probabilities[i,j,k,loop_targets[i,j,k]] > qall_90:
                            q90_labs.append(loop_targets[i,j,k]) 
                            cnt_correct_t[loop_targets[i,j,k], k, t] += 1
                            pv_reg_t[loop_targets[i,j,k], k, t] += pv_anomalies[i,j,t]
                            pv_inp_t[loop_targets[i,j,k], k, t] += pv_inputs[i,j,t] #- avg_spv[loop_targets[i,j,k]]
        pv_anom_reg_t.append(pv_reg_t[None,...]/cnt_correct_t[None,...])
        pv_inp_reg_t.append(pv_inp_t[None,...]/cnt_correct_t[None,...])

    pv_anom_reg_t = np.concatenate(pv_anom_reg_t) # num models x num classes x num output timesteps x num input timesteps
    pv_inp_reg_t = np.concatenate(pv_inp_reg_t)

    pv_anom_reg_mean = np.nanmean(pv_anom_reg_t, axis = 0)
    pv_anom_reg_std = np.nanstd(pv_anom_reg_t, axis = 0)

    cm_list = sns.color_palette("colorblind")
    sns.set_palette("colorblind")

    data_pv = {}
    data_pv['regimes'] = pred_all_90
    data_pv['lead weeks'] = ts_all_90

    pv_strct = {'index':[],'delta t': []}
    delta_t_pv = {reg: pv_strct for reg in regimes}


    for i in range(loop_probabilities.shape[0]):
        for j in range(loop_probabilities.shape[1]):
            for k in range(loop_probabilities.shape[2]):
                for t in range(pv_inputs.shape[2]):
                    if loop_probabilities[i,j,k,loop_targets[i,j,k]] > qall_90: 
                        delta_t = np.abs((t-5)) + (k+1)
                
                        delta_t_pv[regimes[loop_targets[i,j,k]]]['index'].append(pv_inputs[i,j,t])
                        delta_t_pv[regimes[loop_targets[i,j,k]]]['delta t'].append(delta_t)

    pv_anom_tar_dt = []

    for i in range(loop_probabilities.shape[0]): # number models

        pv_tar_dt = np.zeros((4,t_in +t_out-1))
        cnt_tar_dt = np.zeros((4,t_in +t_out-1))
        for j in range(loop_probabilities.shape[1]): # number weeks/ samples
            for k in range(loop_probabilities.shape[2]): # output timesteps
                for t in range(pv_inputs.shape[2]): #input timesteps
                    if not np.isnan(pv_inputs[i,j,t]):
                        d_t = np.abs((t-5)) + (k+1)
                        cnt_tar_dt[loop_targets[i,j,k], d_t-1] += 1
                        pv_tar_dt[loop_targets[i,j,k], d_t-1] += pv_anomalies[i,j,t]

        pv_anom_tar_dt.append(pv_tar_dt[None,...]/cnt_tar_dt[None,...])


    pv_anom_tar_dt = np.concatenate(pv_anom_tar_dt)
    pv_anom_tar_dt_mean = np.nanmean(pv_anom_tar_dt, axis = 0)

    # According to ERA5 predictions.
    pv_anom_all_dt = []

    for i in range(loop_probabilities.shape[0]): # number models
        pv_all_dt = np.zeros((4,t_in +t_out-1))
        cnt_all_dt = np.zeros((4,t_in +t_out-1))
        for j in range(loop_probabilities.shape[1]): # number weeks/ samples
            for k in range(loop_probabilities.shape[2]): # output timesteps
                for t in range(pv_inputs.shape[2]): #input timesteps
                    if not np.isnan(pv_inputs[i,j,t]):
                        d_t = np.abs((t-5)) + (k+1)
                        cnt_all_dt[loop_classes[i,j,k], d_t-1] += 1
                        pv_all_dt[loop_classes[i,j,k], d_t-1] += pv_anomalies[i,j,t]
        pv_anom_all_dt.append(pv_all_dt[None,...]/cnt_all_dt[None,...])


    pv_anom_all_dt = np.concatenate(pv_anom_all_dt)
    pv_anom_all_dt_mean = np.nanmean(pv_anom_all_dt, axis = 0)
    pv_anom_all_dt_std = np.nanstd(pv_anom_all_dt, axis = 0)

    pv_anom_all_t = []


    for i in range(loop_probabilities.shape[0]): # number models
        pv_all_t = np.zeros((4,t_in, t_out))
        cnt_correct_all = np.zeros((4,t_in, t_out))
        for j in range(loop_probabilities.shape[1]): # number weeks/ samples
            for k in range(loop_probabilities.shape[2]): # output timesteps
                for t in range(pv_inputs.shape[2]): #input timesteps
                    if not np.isnan(pv_inputs[i,j,t]):
                        cnt_correct_all[loop_classes[i,j,k], k, t] += 1
                        pv_all_t[loop_classes[i,j,k], k, t] += pv_anomalies[i,j,t]

        pv_anom_all_t.append(pv_all_t[None,...]/cnt_correct_all[None,...])
        

    pv_anom_all_t = np.concatenate(pv_anom_all_t) # num models x num classes x num output timesteps x num input timesteps

    pv_anom_all_mean = np.nanmean(pv_anom_all_t, axis = 0)
    pv_anom_all_std = np.nanstd(pv_anom_all_t, axis = 0)


    pv_all_dt_anom_mean = np.zeros((4,6,t_in +t_out-1))
    pv_all_dt_anom_mean[:] = np.nan
    pv_all_dt_anom_std = np.zeros((4,6,t_in +t_out-1))
    pv_all_dt_anom_std[:] = np.nan
    for reg in range(pv_anom_all_mean.shape[0]):
        for t in range(pv_anom_all_mean.shape[1]):
            dts = np.arange(len(pv_anom_all_mean[reg,t,:]))+t
            pv_all_dt_anom_mean[reg,t,dts] = pv_anom_all_mean[reg,t,:]
            pv_all_dt_anom_std[reg,t,dts] = pv_anom_all_std[reg,t,:]

    pv_all_dt_anom_means = np.nanmean(pv_all_dt_anom_mean, axis = 1)
    pv_all_dt_anom_stds = np.nanstd(pv_all_dt_anom_mean, axis = 1) + np.nanmean(pv_all_dt_anom_std, axis = 1) #error propagation



    cm_list = ['#7fbf7b','#1b7837','#762a83','#9970ab','#c2a5cf'] 

    med_perc = np.nanpercentile(pv_anom_tar_dt[:,:,0].flatten(),30) # upper thresh weak (Tripathi et al. 2015)
    strong_perc = np.nanpercentile(pv_anom_tar_dt[:,:,0].flatten(),80) # lower thresh strong (Tripathi et al. 2015)

    vmax = 52
    delta_t = np.unique(np.array(delta_t_pv[regimes[0]]['delta t']))
    delta_t = np.arange(1,delta_t[-1]+1)
    colors = sns.color_palette("colorblind", n_colors=6)
    alphas = np.linspace(0.5,1, 6)
    marker_list = ['o','s','^','v','D','*']

    np.savez(f'{results_directory}{arch_type}spv_teleconnections.npz', pv_anom_reg_mean = pv_anom_reg_mean, vmax = vmax, strong_perc = strong_perc, med_perc = med_perc, marker_list = marker_list, alphas = alphas,
         delta_t = delta_t, regimes = regimes, pv_anom_tar_dt = pv_anom_tar_dt, pv_anom_tar_dt_mean = pv_anom_tar_dt_mean, pv_anom_all_dt_mean = pv_anom_all_dt_mean, pv_anom_reg_std = pv_anom_reg_std, pv_all_dt_anom_stds = pv_all_dt_anom_stds)
