import argparse
import os
from typing import List
from typing import Optional
import yaml

from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from packaging import version


# DL packages.
import lightning.pytorch as pl
from lightning.pytorch.callbacks import StochasticWeightAveraging, EarlyStopping


import torch
from torch import optim, utils, nn

# HP tuning package.
import optuna

from deepS2S.model import ViTLSTM
from deepS2S.model.loss import FocalLossAdaptive
from deepS2S.dataset.datasets_regimes import TransferData
from deepS2S.utils.utils_build import build_architecture_sweep, build_encoder
from deepS2S.utils.utils_data import cls_weights
from deepS2S.utils.utils_evaluation import evaluate_accuracy
from deepS2S.utils.utils import statics_from_config

def objective_vit(trial: optuna.trial.Trial,
              ) -> float:
    

    
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default='_hps')
    parser.add_argument("--network", type=str, default='conv')
    parser.add_argument("--ntrials", type=int, default=100)
    args = parser.parse_args()

    cfile = args.config
    # Load config and settings.
    exd = os.path.dirname(os.path.abspath(__file__))
    cfd = Path(exd).parent.absolute() 
    config = yaml.load(open(f'{cfd}/config/loop_config{cfile}.yaml'), Loader=yaml.FullLoader)

    config['net_root'] = str(cfd.parent.absolute()) + f'/Data/Network/'
    config['root'] = str(cfd.parent.absolute()) + f'/Data/Network/Sweeps/'
    config['data_root'] = str(cfd.parent.absolute()) + f'/Data'

    strt_yr = config.get('strt','')
    trial_num = config.get('version', '')
    norm_opt = config.get('norm_opt','')
    name_var = config.get('tropics','')

    log_dir = config['net_root'] + f'Sweeps/ViT-LSTM/Sweep_{strt_yr}{trial_num}_{norm_opt}{name_var}/'

    # Initialize optimization range.
    swa =  trial.suggest_float("SWA",1e-5, 1e-1,log=True)
    learning_rate =  trial.suggest_float("learning_rate",1e-4, 1e-1,log=True)
    batch_size = trial.suggest_categorical("batch_size", [36, 72])
    decoder_hidden_dim = trial.suggest_categorical("decoder_hidden_dim", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1,0.9)
    weight_decay = trial.suggest_float("weight_decay",1e-7, 0.5,log=True)
    gamma = {}
    gamma['val'] = 3
    gamma['nurmeric'] = 1
    gc_fine = trial.suggest_float("gradient_clip_fine", 0.,0.9)


    #Initialize static variables.
    device = torch.device("cuda")
    optimizer =  optim.Adam
    if 'calibrated' in norm_opt:
        criterion = FocalLossAdaptive(gamma = gamma['val'], numerical=gamma['nurmeric'], device=device)
    else:
        criterion = nn.CrossEntropyLoss
    var_comb = config['var_comb']

    data_info, seasons = statics_from_config(config)

    pl.seed_everything(42)

    # Create data loader.
    params = {'seasons': seasons, 'test_set_name':config['data'][config['name']]['setname']}
    Fine_data = TransferData(config['data']['dataset_name2'], 
                                var_comb, data_info, batch_size, **params)

    Fine_data.train_dataloader()
    Fine_data.val_dataloader()
    Fine_data.test_dataloader()
    test_loader = Fine_data.test_dataloader()

    
    data = Fine_data.access_dataset()

    cls_wt = cls_weights(data, 'balanced')

    frame_size = [int(x) for x in data['val'][0][0][0].shape][-2:]
    input_dim = [int(x) for x in data['val'][0][0][0].shape][1]
    # remember for reproducibility and analysis
    data_info['var_comb'] = var_comb

    # Build encoder.
    Mae_olr, Mae_u = build_encoder(config)
    Mae_olr = Mae_olr.encoder
    Mae_u = Mae_u.encoder
    enc_path = config['net_root'] + f'MAE/version_{strt_yr}{trial_num}_{norm_opt}/individual_static/'
    config_enc = yaml.load(open(f'{enc_path}config_u.yaml'), Loader=yaml.FullLoader)
    

    # Create model.
    architecture = ViTLSTM.ViT_LSTM

    early_stop_callback = EarlyStopping(monitor="train_acc", 
                                        min_delta=0.00, patience=3, verbose=False, mode="max")

     
    model_params = dict(
        encoder_u = Mae_u,
        encoder_sst = Mae_olr,
        enc_out_shape = [1,config_enc['vit']['dim']],
        in_time_lag=config['data']['n_steps_in'],
        out_time_lag=config['data']['n_steps_out'],
        out_dim=data['val'][0][0][1].shape[-1],
        output_probabilities = True,
        learning_rate =  learning_rate,
        decoder_hidden_dim = decoder_hidden_dim,
        dropout = dropout,
        weight_decay = weight_decay,
        norm_both = trial.suggest_categorical("norm_both", [True, False]),
        norm = trial.suggest_categorical("norm", [True, False]),
        norm_bch = trial.suggest_categorical("norm_bch", [True, False]),
        clbrt = config['network'].get('clbrt',0),
        add_attn = config['network'].get('add_attn',False),
        n_heads = config['network'].get('n_heads',0),
        bs = batch_size,
        gamma = gamma
    )


    trainer_fine = pl.Trainer(
                logger=True,
                gradient_clip_val = gc_fine,
                log_every_n_steps = 5,
                check_val_every_n_epoch=10,
                default_root_dir= log_dir, 
                deterministic=True,
                enable_checkpointing=False,
                max_epochs=config['epochs'],
                accelerator="gpu" ,
                devices=config['devices'],
                strategy='ddp_find_unused_parameters_true',
                callbacks=[StochasticWeightAveraging(swa_lrs=swa), early_stop_callback],
                )
    
    model, exp_info = build_architecture_sweep(name=f'{architecture.__name__}_{"-".join(var_comb["input"])}',
                    architecture = architecture,
                    model_params = model_params,
                    data = data_info,
                    criterion = criterion,
                    optimizer = optimizer,
                    epochs = config['network']['epochs'],
                    class_weight = cls_wt,
                    batch_size=config['data']['bs'],
                    trainer = trainer_fine,
                    n_instances=1)

   
    model_params['swa'] = swa
    model_params['criterion'] = criterion.__name__
    model_params['optimizer'] = optimizer.__name__
    model_params['lr_fine'] = learning_rate
    model_params['swa'] = swa
    model_params['bs'] = batch_size
    model_params['grad_clip_fine'] = gc_fine
    model_params['encoder_u'] = []
    model_params['encoder_sst'] = []

    trainer_fine.logger.log_hyperparams(model_params) #only if trainer logger = False

    trainer_fine.fit(model, Fine_data)

    trainer_fine.save_checkpoint(f"{trainer_fine.logger.log_dir}/best_model.ckpt")
    trainer_fine.test(dataloaders=Fine_data.test_dataloader(), ckpt_path=f"{trainer_fine.logger.log_dir}/best_model.ckpt")

    val_results = trainer_fine.validate(dataloaders=Fine_data.val_dataloader(), ckpt_path=f"{trainer_fine.logger.log_dir}/best_model.ckpt")

    val_ac, _ = evaluate_accuracy(model, trainer_fine, Fine_data.val_dataloader(),
                                    config, data_info, var_comb, seasons, 'val')


    trainer_fine.test(dataloaders=Fine_data.test_dataloader(), ckpt_path=f"{trainer_fine.logger.log_dir}/best_model.ckpt")

    fine_acc, fine_acc_ts = evaluate_accuracy(model, trainer_fine, Fine_data.test_dataloader(),config, data_info, var_comb, seasons, 'test')

    results = {'test_fine': float(fine_acc), 'val_fine': float(val_ac),'val_ece': val_results[0]['val_ece'], 
            'fine_ts_acc': {'ts_acc':fine_acc_ts, 'mean':float(np.mean(fine_acc_ts))}}

    with open(os.path.join(Path(trainer_fine.logger.log_dir), "accuracies.yml"), 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)

    
    return (float(val_ac) - val_results[0]['val_ece'])