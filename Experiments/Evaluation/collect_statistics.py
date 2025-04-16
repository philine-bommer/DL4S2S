# Description: Collects statistics from the model and saves them to a file.
#
# This
# script is used to collect statistics from the model and save them to a file. The
# script loads the model from the specified directory and uses it to make predictions
# on the test data. The predictions are then saved to a file in the specified directory.
# The script also saves the test data and the model parameters to the file.
#
# The script takes the following arguments:
# - config: The configuration file for the experiment.
# - ntrials: The number of trials to run.
#
# The script first loads the configuration file and the test data. It then loops over
# the specified number of trials, loading the model from the specified directory and
# using it to make predictions on the test data. The predictions are saved to a file
# in the specified directory, along with the test data and the model parameters.
#
# The script then saves the results to a file in the specified directory.
#
import os
import yaml
from pathlib import Path
from argparse import ArgumentParser


import xarray as xr
import numpy as np
import pdb

import lightning.pytorch as pl


from deepS2S.model import ViTLSTM
from deepS2S.utils.utils_data import load_data
from deepS2S.utils.utils_evaluation import *
from deepS2S.utils.utils import statics_from_config
from deepS2S.utils.utils_load import collect_statistics_from_file, collect_statistics_from_model
from deepS2S.utils.utils_model import test_model_and_data, best_model_folder
from deepS2S.utils.utils_plot import *


# Set hyperparameters.
parser = ArgumentParser()

parser.add_argument("--config", type=str, default='_lstm')
parser.add_argument("--ntrials", default=100)
args = parser.parse_args()
cfile = args.config

exd = os.path.dirname(os.path.abspath(__file__))
cfd = Path(exd).parent.absolute()
config = yaml.load(open(f'{cfd}/config/config{cfile}.yaml'), Loader=yaml.FullLoader)
config_base = yaml.load(open(f'{cfd}/config/config_vit_lstm.yaml'), Loader=yaml.FullLoader)

strt_yr = config.get('strt','')
trial_num = config.get('version', '')
norm_opt = config.get('norm_opt','')
arch = config.get('arch', 'ViT')
tropics = config.get('tropics', '')
config['net_root'] = str(cfd.parent.absolute()) + f'/Data/Network/'
config['root'] = str(cfd.parent.absolute()) + f'/Data/Network/Sweeps/'
config['data_root'] = str(cfd.parent.absolute()) + f'/Data'

config_base['net_root'] = str(cfd.parent.absolute()) + f'/Data/Network/'
config_base['root'] = str(cfd.parent.absolute()) + f'/Data/Network/Sweeps/'
config_base['data_root'] = str(cfd.parent.absolute()) + f'/Data'



if 'index' == config.get('exp_type', ''):
    stat_dir =  config['net_root'] + f'Statistics/Index_LSTM/'
    result_path = str(cfd.parent.absolute()) + f'/Data/Results/Statistics/Index_LSTM/'
    
elif config['network'].get('mode', 'run') == 'base':
    stat_dir =  config['net_root'] + f'Statistics/LSTM/'
    result_path = str(cfd.parent.absolute()) + f'/Data/Results/Statistics/LSTM/'
else:
    stat_dir =  config['net_root'] + f'Statistics/ViT-LSTM/'
    result_path = str(cfd.parent.absolute()) + f'/Data/Results/Statistics/{arch}/'

results_directory = Path(f'{result_path}version_{strt_yr}{trial_num}_{norm_opt}{tropics}/')
os.makedirs(results_directory, exist_ok=True)

mod_name = 'ViT_LSTM'
architecture = ViTLSTM.ViT_LSTM

if not os.path.exists(results_directory):
        os.makedirs(results_directory)

test_loader, data_set, cls_wt, test_set, infos = load_data(config_base)

var_comb = config_base['var_comb']

data_info, seasons = statics_from_config(config_base)

'''Load initial model'''
params_base = dict( iter = 0)
exp_dir =  f"{stat_dir}version_{strt_yr}{trial_num}_{norm_opt}{tropics}/"
# pdb.set_trace()
bhp_dir, cf_dir, _ = best_model_folder(config, stat_dir, architecture, **params_base)

acc_base, model_baseline, _, _, _, _, _, _ = test_model_and_data(config, bhp_dir, cf_dir, architecture, [1])

trainer = pl.Trainer(accelerator="gpu",
                    devices = [3],
                    max_steps=200, 
                    strategy='ddp_find_unused_parameters_true',
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
olr = []
overall_accuracy = []
error_sum = 0
i = 0

for input, output, weeks, days in data_set:
    targets.append(np.array(output).squeeze())
    dates.append(np.array(weeks).squeeze())
    daytimes.append(np.array(days).squeeze())

    if 'index' == config.get('exp_type', ''):
        u10.append(input[1][None,:,-5:-4].numpy())
        olr.append(input[1][None,:,:-5].numpy())
    else:
        u10.append(input[0][None,:,1,...].numpy())
        olr.append(input[0][None,:,0,...].numpy())


    persistance.append(np.repeat(np.argmax(input[1].numpy()[-1]), 6).T)
    i += 1
    
for var in ['persistance','dates', 'daytimes', 'targets']:
    locals()[var] = np.concatenate(locals()[var]).reshape((predictions_baseline.shape[0],
                                                           predictions_baseline.shape[1]))

for var in ['olr', 'u10']:
    locals()[var] = np.concatenate(locals()[var], axis =0)

params = dict(cls_wgt = cls_wt, sweep =  1)
pths = [xs for xs in Path(exp_dir).iterdir() if xs.is_dir()]

result_data = Path(f'{results_directory}/full_loop_data_{len(pths)}.npz')
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
                  'olr': olr, 
                  'u10': u10, 
                  'dates': dates, 
                  'daytimes': daytimes,
                  'loop_probabilities':loop_probabilities,
                  'loop_classes':loop_classes,
                  'predictions_baseline':predictions_baseline,
                  'targets':targets,}

np.savez(f'{results_directory}/collected_loop_data_{len(pths)-1}.npz',**collected_data)