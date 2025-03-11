import json
import os
import yaml
import importlib
from pathlib import Path
from argparse import ArgumentParser
import shutil
import utils
import pdb

import xarray as xr
import numpy as np




from torch import utils
import lightning.pytorch as pl


from deepS2S.model import ViTLSTM
from deepS2S.utils.utils_data import load_data
import deepS2S.utils.utils_evaluation as eval
from deepS2S.utils.utils import statics_from_config
from deepS2S.utils.utils_load import collect_statistics_from_model, collect_statistics_from_file
from deepS2S.utils.utils_model import test_model_and_data, best_model_folder
from deepS2S.utils.plot_utils import *


# Set hyperparameters.
parser = ArgumentParser()

parser.add_argument("--config", type=str, default='')
parser.add_argument("--ntrials", default=100)
args = parser.parse_args()

cfile = args.config

exd = os.path.dirname(os.path.abspath(__file__))
cfd = exd.parent.absolute()
config = yaml.load(open(f'{cfd}/config/config{cfile}.yaml'), Loader=yaml.FullLoader)
config_base = yaml.load(open(f'{cfd}/config/config_1980_olr.yaml'), Loader=yaml.FullLoader)

strt_yr = config.get('strt','')
trial_num = config.get('version', '')
norm_opt = config.get('norm_opt','')
arch = config.get('arch', 'ViT')
tropics = config.get('tropics', '')
temp_scaling = config.get('temp_scaling', False)

config['net_root'] = str(cfd.parent.absolute()) + f'/Data/Network/'
config['root'] = str(cfd.parent.absolute()) + f'/Data/Network/Sweeps/'
res_path = str(cfd.parent.absolute()) + f'/Data/Results/'

if 'index' == config.get('exp_type', ''):
    stat_dir =  config['net_root'] + f'Statistics/Index_LSTM/'
    result_path = f'{res_path}Statistics/Index_LSTM/'

elif config['network'].get('mode', 'run') == 'base':
    stat_dir =  config['net_root'] + f'Statistics/LSTM/'
    result_path = f'{res_path}Statistics/LSTM/'
else:
    stat_dir =  config['net_root'] + f'Statistics/{arch}'
    result_path = f'{res_path}Statistics/{arch}/'

results_directory = Path(f'{result_path}version_{strt_yr}{trial_num}_{norm_opt}{tropics}/')
os.makedirs(results_directory, exist_ok=True)

base_dir = stat_dir

mod_name = 'ViT_LSTM'
architecture = ViTLSTM.ViT_LSTM

base_name = 'ViT_LSTM'



cm_list = ['#7fbf7b','#1b7837','#762a83','#9970ab','#c2a5cf']  
regimes = ['SB', 'NAO-', 'AR', 'NAO+']

if not os.path.exists(results_directory):
        os.makedirs(results_directory)



test_loader, data_set, cls_wt, test_set, infos = load_data(config_base)

var_comb = config_base['var_comb']

data_info, _ = statics_from_config(config_base)

seasons =  {'train':{config['data']['dataset_name2']:list(range(config['data']['fine']['train_start'], 
                                                                config['data']['fine']['train_end']))},
            'val':{config['data']['dataset_name2']:list(range(config['data']['fine']['val_start'],
                                                               config['data']['fine']['val_end']))},
            'test':{config['data']['dataset_name2']:list(range(config['data']['fine']['test_start'], 
                                                               config['data']['fine']['test_end']))}}

'''Load baseline lstm model'''
params_base = dict( iter = 0)

bhp_dir, cf_dir, _ = best_model_folder(config_base, base_dir, architecture, **params_base)

acc_base, model_baseline, _, _, _, _, _, _ = test_model_and_data(config_base, bhp_dir, cf_dir, architecture, [1])

trainer = pl.Trainer(accelerator="gpu",
                     devices = True,
                    max_steps=200, 
                    default_root_dir= results_directory, 
                    deterministic=True)

result_baseline = trainer.test(model_baseline, dataloaders=test_loader)
pred_baseline = trainer.predict(model_baseline, dataloaders=test_loader)

predictions_baseline = []
for i in range(len(pred_baseline)):
    predictions_baseline.append(pred_baseline[i])

for var in ['predictions_baseline']:
    locals()[var] = np.concatenate(locals()[var])

dates = []
daytimes = []
targets = []

i = 0

for input, output, weeks, days in data_set:
    targets.append(np.array(output).squeeze())
    dates.append(np.array(weeks).squeeze())
    daytimes.append(np.array(days).squeeze())
    i += 1

for var in ['dates', 'daytimes', 'targets']:
    locals()[var] = np.concatenate(locals()[var]).reshape((predictions_baseline.shape[0],
                                                           predictions_baseline.shape[1]))

if 'index' == config.get('exp_type', ''):
    test_loader, data_set, cls_wt, test_set, infos = load_data(config)

    var_comb = config['var_comb']

    data_info, _ = statics_from_config(config)

dates = []
daytimes = []
targets = []
persistance = []
u10 = []
sst = []
overall_accuracy = []
error_sum = 0
i = 0

for input, output, weeks, days in data_set:
    targets.append(np.array(output).squeeze())
    dates.append(np.array(weeks).squeeze())
    daytimes.append(np.array(days).squeeze())

    if 'index' == config.get('exp_type', ''):
        u10.append(input[1][None,:,-5:-4].numpy())
        sst.append(input[1][None,:,:-5].numpy())
    else:
        u10.append(input[0][None,:,1,...].numpy())
        sst.append(input[0][None,:,0,...].numpy())


    persistance.append(np.repeat(np.argmax(input[1].numpy()[-1]), 6).T)
    i += 1
    
for var in ['persistance','dates', 'daytimes', 'targets']:
    locals()[var] = np.concatenate(locals()[var]).reshape((predictions_baseline.shape[0],
                                                           predictions_baseline.shape[1]))

for var in ['sst', 'u10']:
    locals()[var] = np.concatenate(locals()[var], axis =0)

params = dict(cls_wgt = cls_wt, sweep =  1)


exp_dir =  f"{stat_dir}version_{strt_yr}{trial_num}_{norm_opt}{tropics}/"
pths = [xs for xs in Path(exp_dir).iterdir() if xs.is_dir()]

if temp_scaling:
    result_data = Path(f'{results_directory}/full_loop_data_{len(pths)-1}_temp_scale.npz')
    if os.path.exists(result_data):
        res_data = np.load(result_data)
        loop_probabilities = []
        loop_classes = []
        for num in range(1,len(pths)):
            current_dir =  exp_dir + f'run_{num}/'
        
            loop_probabilities.append(res_data[f"run {num}"]['predictions'])
            loop_classes.append(res_data[f"run {num}"]['classes'])
        loop_probabilities = np.concatenate(loop_probabilities).reshape(
                        len(pths),predictions_baseline.shape[0],predictions_baseline.shape[1]
                        ,predictions_baseline.shape[2])
        loop_classes = np.concatenate(loop_classes).reshape(
                        len(pths),predictions_baseline.shape[0],predictions_baseline.shape[1])
    else:
        results, loop_probabilities, loop_classes = collect_statistics_from_file(pths, exp_dir, 
                                                                                 results_directory,)
        np.savez(result_data, **results)

else:
    result_data = Path(f'{results_directory}/full_loop_data_{len(pths)-1}.npz')
    if os.path.exists(result_data):
        res_data = np.load(result_data)
        loop_probabilities = []
        loop_classes = []
        for num in range(1,len(pths)):
            current_dir =  exp_dir + f'run_{num}/'
        
            loop_probabilities.append(res_data[f"run {num}"]['predictions'])
            loop_classes.append(res_data[f"run {num}"]['classes'])
        loop_probabilities = np.concatenate(loop_probabilities).reshape(
                        len(pths),predictions_baseline.shape[0],predictions_baseline.shape[1]
                        ,predictions_baseline.shape[2])
        loop_classes = np.concatenate(loop_classes).reshape(
                        len(pths),predictions_baseline.shape[0],predictions_baseline.shape[1])
    else:
        results, loop_probabilities, loop_classes = collect_statistics_from_model(pths, exp_dir, config, 
                                                                       mod_name, params, trainer, 
                                                                       test_loader, data_info, 
                                                                       var_comb, seasons, 
                                                                       results_directory, targets)
        np.savez(result_data, **results)
collected_data = {'persistance': persistance, 
                  'sst': sst, 
                  'u10': u10, 
                  'dates': dates, 
                  'daytimes': daytimes,
                  'loop_probabilities':loop_probabilities,
                  'loop_classes':loop_classes,
                  'predictions_baseline':predictions_baseline,
                  'targets':targets,}

if temp_scaling:
    np.savez(f'{results_directory}/collected_loop_data_{len(pths)-1}_temp_scale.npz',**collected_data)
else:
    np.savez(f'{results_directory}/collected_loop_data_{len(pths)-1}.npz',**collected_data)