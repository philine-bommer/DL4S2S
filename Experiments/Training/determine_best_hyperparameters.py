import os
import yaml
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd 
import xarray as xr

from torch import utils
import lightning.pytorch as pl

from deepS2S.utils.utils_model import test_model
from deepS2S.model import ViTLSTM


parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument("--config", type=str, default='_hps')
parser.add_argument("--calculate", type=int, default=0)
parser.add_argument("--network", type=str, default='ViT')
parser.add_argument("--ntrials", type=int, default=100)
args = parser.parse_args()

cfile = args.config
ntype = args.network
calc = args.calculate
# Load config and settings.
exd = os.path.dirname(os.path.abspath(__file__))
cfd = Path(exd).parent.absolute() 
config = yaml.load(open(f'{cfd}/config/loop_config{cfile}.yaml'), Loader=yaml.FullLoader)
arch = 'ViT-LSTM/'

config['net_root'] = str(cfd.parent.absolute()) + f'/Data/Network/'
config['root'] = str(cfd.parent.absolute()) + f'/Data/Network/Sweeps/'
config['data_root'] = str(cfd.parent.absolute()) + f'/Data'

if 'Sweep' in config['root']:
    sweep = True
strt_yr = config.get('strt','')
trial_num = config.get('version', '')
norm_opt = config.get('norm_opt','')
name_var = config.get('tropics','')
exp_dir = config['net_root'] + f'Sweeps/{arch}Sweep_{strt_yr}{trial_num}_{norm_opt}{name_var}/'

pl.seed_everything(42)

log_dir = f"{exp_dir}/lightning_logs/"
r = Path(log_dir)
paths = [x for x in r.iterdir() if x.is_dir()]

test_acc = []
val_acc = []
folders = [] 
for pat in paths:
    if calc:
            # mod_name = 'ViT_LSTM'
            mod_name = ViTLSTM.ViT_LSTM
            test_accu, test_acc_ts, val_accu, val_acc_ts = test_model(config, pat, mod_name, [0])
            test_acc.append(np.mean(test_acc_ts))
            
            val_acc.append(np.mean(val_acc_ts))
            folders.append(pat)

    else:
        try: 
            accs = yaml.load(open(pat / 'accuracies.yml'), Loader=yaml.UnsafeLoader)
            if norm_opt:
                test_acc.append(accs['test_fine'])
            else:
                test_acc.append(accs['test_fine'][0]['test_acc'])
            
            val_acc.append(accs['val_fine'])
            folders.append(pat)
        except:
            print(f'{pat} has failed.')


test_acc =np.array(test_acc)
val_acc =np.array(val_acc)

best = np.argmax(test_acc)
print(f'best accuracy: {val_acc[best]}')


result_fl = yaml.load(open(exp_dir + '/results.yml'), Loader=yaml.FullLoader)
if not isinstance(result_fl, dict):
     result_fl = {}


result_fl['test_dir'] = folders[best]
result_fl['test_acc'] = test_acc[best]
result_fl['val_acc'] = val_acc[best]
# pdb.set_trace()
hpms = yaml.load(open(folders[best] / 'hparams.yaml'), Loader=yaml.UnsafeLoader)
result_fl['hparams'] = hpms

with open(exp_dir + '/result.yml', 'w') as outfile:

        yaml.dump(result_fl, outfile, default_flow_style=False)


