import os
from pathlib import Path
import json
import copy
import shutil
from datetime import datetime
from argparse import ArgumentParser

import yaml
import torch
import numpy as np
# import models
import torch.nn as nn
import pdb

import lightning.pytorch as pl


from model import mae, ViT
import model.StNN_static as m
# import model.StNN_static as stnn
from DeepS2S.deepS2S.utils.utils_build import build_encoder
from utils_data import cls_weights
from utils_evaluation import evaluate_accuracy
from utils import statics_from_config
from DeepS2S.deepS2S.dataset.datasets_regimes import TransferData, WeatherDataset


def load_single_model(config, hp_dir, cf_dir, architecture):

    try:
        with open(Path(hp_dir) / 'hparams.yaml', 'r') as f:
            result_collection = yaml.load(f, Loader=yaml.UnsafeLoader)
    except FileNotFoundError:
        result_collection = None

    if 'spatiotemporal' in architecture.__name__:
        # architecture = getattr(stnn, name)
        mae_sst, mae_u = build_encoder(config)
        if "transfer" in config['setting_training']:
            model = architecture.load_from_checkpoint(f"{hp_dir}/best_finetuned_model.ckpt", 
                                                            encoder_u = mae_u.encoder,
                                                            encoder_sst = mae_sst.encoder, )
        else:
            model = architecture.load_from_checkpoint(f"{hp_dir}/best_model.ckpt", 
                                                            encoder_u = mae_u.encoder,
                                                            encoder_sst = mae_sst.encoder, )                                           
                
    else:
        with open(Path(cf_dir) / 'model_architecure.json', 'r') as f:
            exp_info = json.load(f)
        architecture = getattr(m, exp_info['name'])
        if "transfer" in config['setting_training']:
            try:
                model = architecture.load_from_checkpoint(f"{hp_dir}/best_finetuned_model.ckpt",  
                                                        strict=False)
            except:
                model = architecture.load_from_checkpoint(f"{hp_dir}/best_finetuned_model.ckpt", 
                                                    strict=False)
        else:    
            try:
                model = architecture.load_from_checkpoint(f"{hp_dir}/best_model.ckpt", 
                                                        strict=False)
            except:
                model = architecture.load_from_checkpoint(f"{hp_dir}/best_pretrained_model.ckpt", 
                                                    strict=False)
    model.configure_optimizers()
    model.configure_loss()
    
    return model, result_collection, hp_dir, architecture

def get_data(config, seasons, var_comb, data_info):

    # Create data loader.
    params = {'seasons': seasons, 'test_set_name':config['data'][config['name']]['setname']}

    Fine_data = TransferData(config['data']['dataset_name2'], 
                            var_comb, data_info, config['data']['bs'], **params)


    Fine_data.train_dataloader()
    val_loader = Fine_data.val_dataloader()
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


    return test_loader, val_loader, data, cls_wt, test_set, data_set

def test_model(config, pat, architecture, dev):


    model, _, _, _ = load_single_model(config, pat, None, architecture)
    var_comb = config['var_comb']

    data_info, seasons = statics_from_config(config)


    test_loader, val_loader, _, _, _, _ = get_data(config, seasons, var_comb, data_info)

    pl.seed_everything(42)

    trainer = pl.Trainer(logger = False,
                        accelerator="gpu",
                        devices = dev,
                        max_steps=200, 
                        # default_root_dir= results, 
                        deterministic=True)
    
    test_loader, val_loader = get_data(config, seasons, var_comb, data_info)

    test_acc, test_acc_ts = evaluate_accuracy(model, trainer, test_loader,
                                              config, data_info, var_comb, seasons, 'test')
    val_acc, val_acc_ts = evaluate_accuracy(model, trainer, val_loader,
                                            config, data_info, var_comb, seasons, 'val')

     
    return test_acc, test_acc_ts, val_acc, val_acc_ts

def test_model_and_data(config, pat, dpat, architecture, dev):


    model, _, _, _ = load_single_model(config, pat, dpat, architecture)
    var_comb = config['var_comb']

    data_info, seasons = statics_from_config(config)
        
    infos = [data_info, var_comb, seasons]
        
    # Create data loader.
    test_loader, val_loader, data, cls_wt, test_set, data_set = get_data(config, seasons, var_comb, data_info)

    trainer = pl.Trainer(logger = False,
                        accelerator="gpu",
                        devices = dev,
                        max_steps=200, 
                        # default_root_dir= results, 
                        deterministic=True)


    test_acc, test_acc_ts = evaluate_accuracy(model, trainer, test_loader,
                                              config, data_info, var_comb, seasons, 'test')
    val_acc, val_acc_ts = evaluate_accuracy(model, trainer, val_loader,
                                            config, data_info, var_comb, seasons, 'val')
    
    acc = [test_acc, test_acc_ts, val_acc, val_acc_ts]

     
    return acc, model, test_loader, val_loader, data_set, cls_wt, test_set, infos

def best_model_folder(config, exp_dir, architecture, **params):

    name = architecture.__name__

    if params.get("sweep",0):
        name_var = config.get('tropics','')   

        exp_dir =  exp_dir + config['strt'] + config['version'] + '_' + config.get('norm_opt','') + name_var + '/'
        print(exp_dir)
        try : 
            con_res = yaml.load(open(exp_dir + 'result.yml'), Loader=yaml.UnsafeLoader)
        except: 
            exp_dir = config['root']
            con_res = yaml.load(open(exp_dir + 'result.yml'), Loader=yaml.UnsafeLoader)

        log_dir = f"{exp_dir}lightning_logs/"
        hp_dir = log_dir + str(con_res.get("test_dir",0).stem)
        cf_dir = hp_dir

    else:
        strt_yr = config.get('strt','')
        trial_num = config.get('version', '')
        norm_opt = config.get('norm_opt','')
        log_dir = exp_dir + f'version_{strt_yr}{trial_num}_{norm_opt}/'
        if not os.path.exists(log_dir):
            log_dir = exp_dir + 'lightning_logs/'

            r = Path(log_dir)
            paths = [x for x in r.iterdir() if x.is_dir()]
            if len(paths) == 1:
                hp_dir = paths[0]
            else:
                try:
                    hp_dir = Path(log_dir + f'version_{params.get("iter_dir",0)}/')
                except:
                    hp_dir = paths[params.get("iter",0)]
            cf_dir = hp_dir
  
        else:
            cfg_dir = log_dir
            dr = Path(cfg_dir)
            pths = [xs for xs in dr.iterdir() if name in xs.stem]
            cf_dir = pths[0]
            del dr, pths
            log_dir = log_dir + 'lightning_logs/'
            hp_dir = Path(log_dir + 'version_0/')
    
    print(f'Analyzing model in {hp_dir}')

    try:
        with open(Path(hp_dir) / 'hparams.yaml', 'r') as f:
            result_collection = yaml.load(f, Loader=yaml.UnsafeLoader)
    except FileNotFoundError:
        print("File not found.")
        result_collection = None

    return hp_dir, cf_dir, result_collection

def set_temperature(model, valid_loader):
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            logits = model(inputs[0], inputs[1])
            logits_list.append(logits)
            labels_list.append(labels)
   
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    temperature_scaling = m.TemperatureScaling()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS([temperature_scaling.temperature], lr=0.01, max_iter=50)

    def eval():
        loss = criterion(temperature_scaling(logits).reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()
        return loss

    optimizer.step(eval)

    return temperature_scaling
