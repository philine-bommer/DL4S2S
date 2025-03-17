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


    lp_pr = loop_probabilities.reshape(loop_classes.shape[0],loop_classes.shape[1]*loop_classes.shape[2],
                                    loop_probabilities.shape[3])
    loop_targets = np.repeat(targets[None,:,:], loop_classes.shape[0], axis = 0)
    lp_conf = loop_probabilities.flatten()
    q_90 = 90
    qall_90 = np.percentile(lp_conf,q_90)


    # Build loop baselines
    loop_tgs = loop_targets.reshape(loop_classes.shape[0]*loop_classes.shape[1],loop_classes.shape[2])
    loop_cls = loop_classes.reshape(loop_classes.shape[0]*loop_classes.shape[1],loop_classes.shape[2])
    loop_prbs = loop_probabilities.reshape(loop_classes.shape[0]*loop_classes.shape[1],loop_classes.shape[2],loop_probabilities.shape[3])


    index_path = str(cfd.parent.absolute()) + f'/Data/Index'
    rmm1_index = np.load(f'{index_path}/rmm1_index_1980-2023_mjo_testset.npz')
    rmm2_index = np.load(f'{index_path}/rmm2_index_1980-2023_mjo_testset.npz')

    rmm1_index_in = rmm1_index['input']
    rmm1_index_out = rmm1_index['output']
    daytimes = rmm1_index['daytimes']
    dates = rmm1_index['dates']

    rmm2_index_in = rmm2_index['input']
    rmm2_index_out = rmm2_index['output']
    rmm2_inputs = np.repeat(rmm2_index_in[None,:,:], loop_classes.shape[0], axis = 0)
    rmm2_outputs = np.repeat(rmm2_index_out[None,:,:], loop_classes.shape[0], axis = 0)
    rmm1_inputs = np.repeat(rmm1_index_in[None,:,:], loop_classes.shape[0], axis = 0)
    rmm1_outputs = np.repeat(rmm1_index_out[None,:,:], loop_classes.shape[0], axis = 0)

    sum_rmm1_reg = np.zeros((4,1))
    sum_rmm2_reg = np.zeros((4,1))
    cnts_reg_in  = np.zeros((4,1))


    for i in range(loop_probabilities.shape[0]):
        for j in range(loop_probabilities.shape[1]):
            for k in range(loop_probabilities.shape[2]):
                for t in range(rmm1_inputs.shape[2]):
                    if not np.isnan(rmm1_inputs[i,j,t]):
                        cnts_reg_in[input_reg[j,k]] += 1
                        sum_rmm1_reg[input_reg[j,k]] += rmm1_inputs[i,j,t]
                        sum_rmm2_reg[input_reg[j,k]] += rmm2_inputs[i,j,t]

    avg_rmm1 = (sum_rmm1_reg.T/cnts_reg_in).flatten()
    avg_rmm2 = (sum_rmm2_reg.T/cnts_reg_in).flatten()


    rmm1_anomalies = np.zeros(rmm1_inputs.shape)
    rmm2_anomalies = np.zeros(rmm1_inputs.shape)

    for i in range(loop_probabilities.shape[0]):
        for j in range(loop_probabilities.shape[1]):
            for k in range(loop_probabilities.shape[2]):
                for t in range(rmm1_inputs.shape[2]):
                        if not np.isnan(rmm1_inputs[i,j,t]):
                            # d_t = np.abs((t-5)) + (k+1)
                            rmm1_anomalies[i,j,t] = rmm1_inputs[i,j,t]- avg_rmm1[input_reg[j,k]]
                            rmm2_anomalies[i,j,t] = rmm2_inputs[i,j,t]- avg_rmm2[input_reg[j,k]]

    rmm1_anom_reg_t = []
    rmm2_anom_reg_t = []

    t_in, t_out = loop_classes.shape[2], loop_probabilities.shape[2]

    for i in range(loop_probabilities.shape[0]): # number models
        rmm1_reg_t = np.zeros((4,t_in, t_out))
        rmm2_reg_t = np.zeros((4,t_in, t_out))
        cnt_correct_t = np.zeros((4,t_in, t_out))
        for j in range(loop_probabilities.shape[1]): # number weeks/ samples
            for k in range(loop_probabilities.shape[2]): # output timesteps
                for t in range(rmm1_inputs.shape[2]): #input timesteps
                    if not np.isnan(rmm1_inputs[i,j,t]):
                        if loop_probabilities[i,j,k,loop_targets[i,j,k]] > qall_90: 
                            cnt_correct_t[loop_targets[i,j,k], k, t] += 1
                            rmm1_reg_t[loop_targets[i,j,k], k, t] += rmm1_anomalies[i,j,t]
                            rmm2_reg_t[loop_targets[i,j,k], k, t] += rmm2_anomalies[i,j,t]
        rmm1_anom_reg_t.append(rmm1_reg_t[None,...]/cnt_correct_t[None,...])
        rmm2_anom_reg_t.append(rmm2_reg_t[None,...]/cnt_correct_t[None,...])

    rmm1_anom_reg_t = np.concatenate(rmm1_anom_reg_t) # num models x num classes x num output timesteps x num input timesteps
    rmm2_anom_reg_t = np.concatenate(rmm2_anom_reg_t) # num models x num classes x num output timesteps x num input timesteps

    rmm1_anom_reg_mean = np.nanmean(rmm1_anom_reg_t, axis = 0)
    rmm1_anom_reg_std = np.nanstd(rmm1_anom_reg_t, axis = 0)
    rmm2_anom_reg_mean = np.nanmean(rmm2_anom_reg_t, axis = 0)
    rmm2_anom_reg_std = np.nanstd(rmm1_anom_reg_t, axis = 0)

    rmm1_anom_tar_dt = []
    rmm2_anom_tar_dt = []

    for i in range(loop_probabilities.shape[0]): # number models
        rmm1_tar_dt = np.zeros((4,t_in +t_out-1))
        rmm2_tar_dt = np.zeros((4,t_in +t_out-1))
        cnt_tar_dt = np.zeros((4,t_in +t_out-1))
        for j in range(loop_probabilities.shape[1]): # number weeks/ samples
            for k in range(loop_probabilities.shape[2]): # output timesteps
                for t in range(rmm1_inputs.shape[2]): #input timesteps
                    if not np.isnan(rmm1_inputs[i,j,t]):
                        d_t = np.abs((t-5)) + (k+1)
                        cnt_tar_dt[loop_targets[i,j,k], d_t-1] += 1
                        rmm1_tar_dt[loop_targets[i,j,k], d_t-1] += rmm1_anomalies[i,j,t]
                        rmm2_tar_dt[loop_targets[i,j,k], d_t-1] += rmm2_anomalies[i,j,t]

        rmm1_anom_tar_dt.append(rmm1_tar_dt[None,...]/cnt_tar_dt[None,...])
        rmm2_anom_tar_dt.append(rmm2_tar_dt[None,...]/cnt_tar_dt[None,...])

    rmm1_anom_tar_dt = np.concatenate(rmm1_anom_tar_dt)
    rmm2_anom_tar_dt = np.concatenate(rmm2_anom_tar_dt)

    rmm1_anom_tar_dt_mean = np.nanmean(rmm1_anom_tar_dt, axis = 0)
    rmm1_anom_tar_dt_std = np.nanstd(rmm1_anom_tar_dt, axis = 0)
    rmm2_anom_tar_dt_mean = np.nanmean(rmm2_anom_tar_dt, axis = 0)
    rmm2_anom_tar_dt_std = np.nanstd(rmm2_anom_tar_dt, axis = 0)

    rmm1_anom_all_dt = []
    rmm2_anom_all_dt = []

    for i in range(loop_probabilities.shape[0]): # number models

        rmm1_all_dt = np.zeros((4,t_in +t_out-1))
        rmm2_all_dt = np.zeros((4,t_in +t_out-1))
        cnt_all_dt = np.zeros((4,t_in +t_out-1))
        for j in range(loop_probabilities.shape[1]): # number weeks/ samples
            for k in range(loop_probabilities.shape[2]): # output timesteps
                for t in range(rmm1_inputs.shape[2]): #input timesteps
                    if not np.isnan(rmm1_inputs[i,j,t]):
                        d_t = np.abs((t-5)) + (k+1)
                        cnt_all_dt[loop_classes[i,j,k], d_t-1] += 1
                        rmm1_all_dt[loop_classes[i,j,k], d_t-1] += rmm1_anomalies[i,j,t]
                        rmm2_all_dt[loop_classes[i,j,k], d_t-1] += rmm2_anomalies[i,j,t]

        rmm1_anom_all_dt.append(rmm1_all_dt[None,...]/cnt_all_dt[None,...])
        rmm2_anom_all_dt.append(rmm2_all_dt[None,...]/cnt_all_dt[None,...])

    rmm1_anom_all_dt = np.concatenate(rmm1_anom_all_dt)
    rmm2_anom_all_dt = np.concatenate(rmm2_anom_all_dt)
    rmm1_anom_all_dt_mean = np.nanmean(rmm1_anom_all_dt, axis = 0)
    rmm1_anom_all_dt_std = np.nanstd(rmm1_anom_all_dt, axis = 0)
    rmm2_anom_all_dt_mean = np.nanmean(rmm2_anom_all_dt, axis = 0)
    rmm2_anom_all_dt_std = np.nanstd(rmm2_anom_all_dt, axis = 0)
    ##1
    index_path = str(cfd).parent.absolute() + f'/Index'
    mjo_index = np.load(f'{index_path}/MJO_index_1980-2023_mjo_testset.npz')


    mjo_index_in = mjo_index['input']
    mjo_cat9_in = np.repeat(mjo_index_in[None,:,:], loop_classes.shape[0], axis = 0)

    inactive_active_mjo = np.zeros((loop_probabilities.shape[0],2,2)) # inactive, active, correct, incorrect

    # conditional probabilities.
    t_in, t_out = mjo_cat9_in.shape[2],loop_probabilities.shape[2]
    for mod in range(loop_probabilities.shape[0]): 
        sub_mjo = mjo_cat9_in[mod,...]
        sub_cl = loop_classes[mod,...]
        sub_targets = loop_targets[mod,...]


        for j in range(sub_cl.shape[0]):
            for k in range(sub_cl.shape[1]):
                for t in range(rmm1_inputs.shape[2]):
                    if not np.isnan(rmm1_inputs[i,j,k]):
                        if sub_cl[j,k] == sub_targets[j,k]:
                            if sub_mjo[j,t] == 0:
                                inactive_active_mjo[mod,0,0] += 1
                            else:
                                inactive_active_mjo[mod,0,1] += 1
                        else:
                            if sub_mjo[j,t] == 0:
                                inactive_active_mjo[mod,1,0] += 1
                            else:
                                inactive_active_mjo[mod,1,1] += 1


    inactive_active_mjo = (inactive_active_mjo/mjo_cat9_in.shape[2])/sub_cl.shape[1]
    mean_inact_act = np.mean(inactive_active_mjo, axis = 0)
    std_inact_act = np.std(inactive_active_mjo, axis = 0)

    vmax_rmm1 = 2.5
    vmax_rmm2 = 2.5
    cm_list = ['#7fbf7b','#1b7837','#762a83','#9970ab','#c2a5cf'] 
    colors = sns.color_palette("colorblind", n_colors=6)
    alphas = np.linspace(0.2,1, 6)
    alphas_dt = np.linspace(0.2,1, rmm1_anom_all_dt_mean[r,:].shape[0])
    marker_list = ['o','s','^','v','D','*']

    amplitude_threshold = np.nanmean(rmm1_anom_reg_mean) + np.nanstd(rmm1_anom_reg_mean) # 1 std above the mean of the anomalies

    np.savez(f'{results_directory}{arch_type}mjo_teleconnections.npz', rmm1_anom_reg_mean = rmm1_anom_reg_mean, rmm2_anom_reg_mean = rmm2_anom_reg_mean,
        rmm1_anom_reg_std = rmm1_anom_reg_std, rmm2_anom_reg_std = rmm2_anom_reg_std, rmm1_anom_all_dt_mean = rmm1_anom_all_dt_mean, rmm2_anom_all_dt_mean = rmm2_anom_all_dt_mean,
        rmm1_anom_tar_dt_mean = rmm1_anom_tar_dt_mean, rmm2_anom_tar_dt_mean = rmm2_anom_tar_dt_mean, vmax = 2.5, alphas = alphas, alphas_dt = alphas_dt,
        vmax_rmm2= vmax_rmm2, vmax_rmm1 = vmax_rmm1, regimes = regimes, amplitude_threshold = amplitude_threshold, cm_list = cm_list, 
        colors = colors, marker_list = marker_list, mean_inact_act = mean_inact_act, std_inact_act = std_inact_act)