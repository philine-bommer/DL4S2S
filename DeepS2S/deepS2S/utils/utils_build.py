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
import pdb
from ..model import mae, ViT, ViTLSTM, IndexLSTM
from .utils_data import cls_weights


def build_architecture(name,
                          architecture,
                          model_params,
                          data,
                          args,
                          criterion,
                          optimizer,
                          epochs,
                          class_weight,
                          batch_size,
                          n_instances=1,
                          target_dir = None,
                        #   print_freq=50,
                        #   initial_weights=None,
                          ):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Computations running on {device}')

    # create folder for experiment
    if 'baseline' in name:
        # target_dir = f'{target_dir}{name}_{datetime.now().strftime("%Y-%m-%dT%H-%M")}'
        mp = model_params
        model_params['optimizer'] = [] 
        mp = copy.deepcopy(model_params)
        model_params['optimizer'] = optimizer
        model_params['cls_wgt'] = class_weight
        model_params['criterion'] = criterion
    else:
        # target_dir = f'{target_dir}{name}_{datetime.now().strftime("%Y-%m-%dT%H-%M")}'

        mp = model_params
        if args:
            model_params['hidden_dim'] = args.hidden_dim
            model_params['encoder_num_layers'] = args.encoder_num_layers
            model_params['decoder_hidden_dim'] = args.decoder_hidden_dim
            model_params['maxpool_kernel_size'] = args.maxpool_kernel_size
            model_params['dropout'] = args.dropout
            model_params['learning_rate'] = args.learning_rate
            # model_params['criterion'] = criterion
            model_params['weight_decay'] = args.weight_decay
        model_params['optim'] = [] 
  
        if model_params.get('encoder_sst',0) or model_params.get('encoder_u',0):
            encoder = model_params['encoder_u']
            encoder_D = model_params['encoder_sst']
            model_params['encoder_u'] = [] 
            model_params['encoder_sst'] = [] 
        else:
            encoder = None
            encoder_D = None
        mp = copy.deepcopy(model_params)
        model_params['optim'] = optimizer
        model_params['cls_wgt'] = class_weight
        model_params['criterion'] = criterion
        
    # if not os.path.exists(target_dir):
    #     os.makedirs(target_dir)




    # store info on model architecure:current model.py file, str(model) as txt
   
    experiment_info = {
        'name': architecture.__name__,
        'architecture': str(architecture(**model_params)),
        'optimizer': optimizer.__name__,
        'batch_size': batch_size,
        'n_epochs': epochs,
        'n_instances': n_instances,
        'data_info': data,
        'model_parameters': mp
    }
    try:
        experiment_info['criterion'] = criterion.__name__
    except:
        experiment_info['criterion'] = 'cross_entropy'
        
    if args:
        experiment_info['learning_rate'] = str(args.learning_rate)
        experiment_info['weight_decay'] = str(args.weight_decay)
        experiment_info['dropout'] = str(args.dropout)

    if encoder_D or encoder:
        model_params['encoder_u'] = encoder
        model_params['encoder_sst'] = encoder_D
    model = architecture(**model_params)
    model.configure_optimizers()
    model.configure_loss()

    if encoder_D or encoder:
        model.freeze_encoder()


    
    return model, experiment_info

def build_finetune(data,
                   architecture, 
                   trainer,
                   **kwargs):

    class_weight = cls_weights(data.access_dataset(), 'balanced')
    kwargs['cls_wgt'] = class_weight
    if not kwargs.get('criterion', None):
        kwargs['criterion'] = torch.nn.CrossEntropyLoss
    
    model = architecture.load_from_checkpoint(f"{trainer.logger.log_dir}/best_pretrained_model.ckpt", **kwargs,strict=False)
    model.configure_optimizers()
    model.configure_loss()
    if kwargs.get('encoder_sst',0) or kwargs.get('encoder_u',0):
        model.freeze_encoder()
    # model.load_from_checkpoint(f"{trainer.logger.log_dir}/best_pretrained_model.ckpt")

    return model 

def build_architecture_sweep(name,
                          architecture,
                          model_params,
                          data,
                          criterion,
                          optimizer,
                          epochs,
                          class_weight,
                          batch_size,
                          trainer,
                          n_instances=1,
                          ):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Computations running on {device}')


    target_dir = f'{trainer.logger.log_dir}/{name}/{name}_{datetime.now().strftime("%Y-%m-%dT%H-%M")}'

    if model_params.get('encoder_sst',0) and model_params.get('encoder_u',0):
        encoder = model_params['encoder_u']
        encoder_D = model_params['encoder_sst']
        model_params['encoder_u'] = [] 
        model_params['encoder_sst'] = [] 
    # else:
    # model_params['hidden_dim'] = model_params['hidden_dim']
    # model_params['encoder_num_layers'] = model_params['encoder_num_layers']
    # model_params['decoder_hidden_dim'] = model_params['decoder_hidden_dim']
    # model_params['maxpool_kernel_size'] = model_params['maxpool_kernel_size']
    # model_params['dropout'] = model_params['dropout']
    # model_params['learning_rate'] = model_params['learning_rate']
    # # model_params['criterion'] = criterion
    # model_params['weight_decay'] = model_params['weight_decay']
    model_params['optim'] = []
    model_params['gamma']= []
    mp = copy.deepcopy(model_params)
    model_params['optim'] = optimizer
    model_params['cls_wgt'] = class_weight
    model_params['criterion'] = criterion
        
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)




    # store info on model architecure:current model.py file, str(model) as txt
   
    experiment_info = {
        'name': architecture.__name__,
        'architecture': str(architecture(**model_params)),
        'optimizer': optimizer.__name__,
        'batch_size': batch_size,
        'criterion': criterion.__name__,
        'n_epochs': epochs,
        'n_instances': n_instances,
        'data_info': data,
        'model_parameters': mp
    }

    with open(os.path.join(target_dir, 'model_architecure.json'), 'w+') as f:
        json.dump(experiment_info, f, indent=4)
    
    if encoder_D or encoder:
        model_params['encoder_u'] = encoder
        model_params['encoder_sst'] = encoder_D
    model = architecture(**model_params)
    model.configure_optimizers()
    model.configure_loss()
    if encoder_D or encoder:
        model.freeze_encoder()

    
    return model, experiment_info

def build_encoder(config):

    # Build encoder.
    strt_yr = config.get('strt','')
    trial_num = config.get('version', '')
    norm_opt = config.get('norm_opt','')
    enc_type = config.get('enc','')
    enc_path = config['net_root'] + f'MAE/version_{strt_yr}{trial_num}_{norm_opt}/individual{enc_type}/'

    
    config_enc = yaml.load(open(f'{enc_path}config_u.yaml'), Loader=yaml.FullLoader)
    encoder = ViT(image_size = tuple(config_enc['loaded_pars']['frame_size']),
                      patch_size = tuple(config_enc['vit']['patch_size']),
                      num_classes = config_enc['vit']['num_classes'],
                      dim = config_enc['vit']['dim'],
                      depth = config_enc['vit']['depth'],
                      heads = config_enc['vit']['heads'],
                      mlp_dim = config_enc['vit']['mlp_dim'],
                      dropout = config_enc['vit']['dropout'],
                      emb_dropout = config_enc['vit']['emb_dropout'],
                      channels = config_enc['vit']['channels'])


    Mae_sst = mae.MAE(
        encoder = encoder,
        masking_ratio = config_enc['mae']['masking_ratio'],   # the paper recommended 75% masked patches
        decoder_dim = config_enc['mae']['decoder_dim'],      # paper showed good results with just 512
        decoder_depth = config_enc['mae']['decoder_depth']       # anywhere from 1 to 8
    )

    Mae_u = mae.MAE(
        encoder = encoder,
        masking_ratio = config_enc['mae']['masking_ratio'],   # the paper recommended 75% masked patches
        decoder_dim = config_enc['mae']['decoder_dim'],      # paper showed good results with just 512
        decoder_depth = config_enc['mae']['decoder_depth']       # anywhere from 1 to 8
    )
    
    tropic_folder = config['tropic_folder']
    u_folder = config['u_folder']
    checkpoint_sst = torch.load(f"{enc_path}lightning_logs/{tropic_folder}.ckpt", map_location=lambda storage, loc: storage)
    Mae_sst.load_state_dict(checkpoint_sst['state_dict'])

    checkpoint_u = torch.load(f"{enc_path}lightning_logs/{u_folder}.ckpt", map_location=lambda storage, loc: storage)
    Mae_u.load_state_dict(checkpoint_u['state_dict'])

    return Mae_sst, Mae_u

def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def load_model(config, exp_dir, name, **params):

    if params.get("sweep",0):
        exp_dir =  exp_dir + config['strt'] + config['version'] + '_' + config.get('norm_opt','') + '/'
        print(exp_dir)
        try : 
            con_res = yaml.load(open(exp_dir + 'result.yml'), Loader=yaml.UnsafeLoader)
        except: 
            exp_dir = config['root']
            con_res = yaml.load(open(exp_dir + 'result.yml'), Loader=yaml.UnsafeLoader)

        log_dir = f"{exp_dir}lightning_logs/"
        hp_dir = log_dir + str(con_res.get("test_dir",0).stem)

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
        result_collection = None



    architecture = getattr(ViTLSTM, name)
    mae_sst, mae_u = build_encoder(config)
    model = architecture.load_from_checkpoint(f"{hp_dir}/best_finetuned_model.ckpt", 
                                                    encoder_u = mae_u.encoder,
                                                        encoder_sst = mae_sst.encoder, )
                                                    
    model.configure_optimizers()
    model.configure_loss()
    # print(model)
    
    
    return model, result_collection, hp_dir, architecture

def load_multi_model(config, current_dir, name, **params):

    dir = current_dir
   
    log_dir = dir + 'lightning_logs/'
    hp_dirs = [xs for xs in Path(log_dir).iterdir() if xs.is_dir()]
    if len(hp_dirs)== 1:
        hp_dir = Path(hp_dirs[0])
    else: 
        hp_dir = Path(log_dir + 'version_1')
    cf_dir = hp_dir

    
    print(f'Analyzing model in {hp_dir}')

    try:
        with open(Path(hp_dir) / 'hparams.yaml', 'r') as f:
            result_collection = yaml.load(f, Loader=yaml.UnsafeLoader)
    except FileNotFoundError:
        result_collection = None

    if "Index_LSTM" in log_dir:
        architecture = getattr(IndexLSTM, name)

    else:
        architecture = getattr(ViTLSTM, name)

    mae_sst, mae_u = build_encoder(config)
    if 'index' in config['var_comb']['input'][0]:
        mae_sst = None
    else:
        mae_sst = mae_sst.encoder

    if 'index' in config['var_comb']['input'][1]:
        mae_u = None
    else:
        mae_u = mae_u.encoder

    model = architecture.load_from_checkpoint(f"{hp_dir}/best_model.ckpt", 
                                                    encoder_u = mae_u,
                                                    encoder_sst = mae_sst, )
                                                    
                
    model.configure_optimizers()
    model.configure_loss()
    # print(model)
    
    results = np.load(f"{dir}predictions.npz")
    
    return model, results, hp_dir, architecture
# def load_model(config, exp_dir, name, architecture, **params):

 
#     strt_yr = config.get('strt','')
#     trial_num = config.get('version', '')
#     norm_opt = config.get('norm_opt','')
#     log_dir = exp_dir + f'version_{strt_yr}{trial_num}_{norm_opt}/'

#     cfg_dir = log_dir + f'{name}/'
#     log_dir = log_dir + 'lightning_logs/'
#     r = Path(log_dir)
#     paths = [x for x in r.iterdir() if params.get("iter",0) in str(x)]
#     if len(paths) == 1:
#         hp_dir = paths[0]
#     else:
#         print(paths)

#     cf_dir = Path(cfg_dir)
 
    
#     print(f'Analyzing model in {hp_dir}')

#     try:
#         with open(Path(hp_dir) / 'hparams.yaml', 'r') as f:
#             result_collection = yaml.load(f, Loader=yaml.UnsafeLoader)
#     except FileNotFoundError:
#         result_collection = None


#     with open(Path(cf_dir) / 'model_architecure.json', 'r') as f:
#         exp_info = json.load(f)

#     cls_wgt_hp = result_collection['cls_wgt']
#     mae_sst, mae_u = build_encoder(config)

#     model = architecture.load_from_checkpoint(f"{hp_dir}/best_model.ckpt", 
#                                                     encoder_u = mae_u,
#                                                     encoder_sst = mae_sst,)  
#                                                     # cls_wgt = cls_wgt_hp,#params['cls_wgt'], 
#                                                     # strict=False)
#     checkpoint = torch.load(f"{hp_dir}/best_model.ckpt")
#     model = weights_update(model = model, checkpoint = checkpoint)
#     # model.load_state_dict(checkpoint['state_dict'])

#     # model.configure_optimizers()
#     # model.configure_loss()
#     # print(model)
    
    
#     return model, result_collection, hp_dir

