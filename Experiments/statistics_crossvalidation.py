import os
import yaml
from argparse import ArgumentParser
import shutil

import torch
import pdb
from torch import dropout, nn, utils
import torch.optim as optim
import numpy as np

# DL packages.
import lightning
import lightning.pytorch as pl
from lightning.pytorch.callbacks import StochasticWeightAveraging, EarlyStopping

from model import mae, ViT, StNN_static
from loss import FocalLossAdaptive
from dataset.datasets_wrapped import TransferData
from build_model import build_architecture, build_finetune, build_encoder
from utils_data import cls_weights
from utils_evaluation import evaluate_accuracy, numpy_predict
from utils import statics_from_config, get_params_from_best_model, get_random_seasons

if __name__ == '__main__':

    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--notification_email", type=str, default="pbommer@atb-potsdam.de")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default=1)
    parser.add_argument("--config", type=str, default='')
    parser.add_argument("--network", type=str, default='ViT')
    parser.add_argument("--ntrials", type=int, default=20)

    args = parser.parse_args()

    cfile = args.config

    # Load config and settings.
    cfd = os.path.dirname(os.path.abspath(__file__))
    config = yaml.load(open(f'{cfd}/config/loop_config{cfile}.yaml'), Loader=yaml.FullLoader)

    num_mods = args.ntrials

    args = parser.parse_args()

    log_dir = config['net_root'] + 'Statistics/Cross-Validation/' 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    strt_yr = config.get('strt','')
    trial_num = config.get('version', '')
    norm_opt = config.get('norm_opt','')
    name_var = config.get('tropics','')
    log_dirs = log_dir + f'version_{strt_yr}{trial_num}_{norm_opt}{name_var}/'
    if not os.path.exists(log_dirs):
        os.makedirs(log_dirs)

    early_stop_callback = EarlyStopping(monitor="val_acc", 
                                        min_delta=0.00, patience=3, verbose=False, mode="max")

    device = torch.device("cuda")


    # Build encoder.
    Mae_sst, Mae_u = build_encoder(config)
    enc_path = config['net_root'] + f'MAE/version_{strt_yr}{trial_num}_{norm_opt}/individual_static/'
    config_enc = yaml.load(open(f'{enc_path}config_u.yaml'), Loader=yaml.FullLoader)
    Mae_sst = Mae_sst.encoder
    Mae_u = Mae_u.encoder

    # Set up models args.
    ntype = args.network
    config['root'] = config['root'] + f"{ntype}/"
    architecture = StNN_static.spatiotemporal_Neural_Network
    conv_params = get_params_from_best_model(config, 'spatiotemporal')

    data_info, _ = statics_from_config(config)
    var_comb = config['var_comb']
    data_info['var_comb'] = var_comb

    #Initialize static variables.
    setting_training = config['setting_training']


    if 'calibrated'in norm_opt:
        criterion = FocalLossAdaptive(gamma = conv_params['gamma'].get('val',3), 
                                                          numerical = conv_params['gamma'].get('numeric',False), 
                                                          device = device)



    pl.seed_everything(1)

    for i in range(num_mods):
    
        train_indices, val_indices, test_indices = get_random_seasons(config['data']['fine']['test_end'], [0.8,0.2])
        seasons =  {'train':{config['data']['dataset_name2']:train_indices},
                'val':{config['data']['dataset_name2']:val_indices},
                'test':{config['data']['dataset_name2']:test_indices}}

        # Create data loader.
        params = {'seasons': seasons, 'test_set_name':config['data'][config['name']]['setname']}

        Fine_data = TransferData(config['data']['dataset_name2'], 
                                    var_comb, data_info, conv_params['bs'], **params)

        Fine_data.train_dataloader()
        Fine_data.val_dataloader()
        Fine_data.test_dataloader()
        test_loader = Fine_data.test_dataloader()

        
        data = Fine_data.access_dataset()

        cls_wt = cls_weights(data, 'balanced')

    
        frame_size = [int(x) for x in data['val'][0][0][0].shape][-2:]
        input_dim = [int(x) for x in data['val'][0][0][0].shape][1]
        # remember for reproducibility and analysis
        
        
        # Build model
        model_params = dict(
            encoder_u = Mae_u,
            encoder_sst = Mae_sst,
            enc_out_shape = [1,config_enc['vit']['dim']],#config_enc['enc_shape'],
            in_time_lag=config['data']['n_steps_in'],
            out_time_lag=config['data']['n_steps_out'],
            out_dim=data['val'][0][0][1].shape[-1],
            output_probabilities=True,
            norm_both = conv_params['norm_both'],
            weight_decay = conv_params['weight_decay'],
            decoder_hidden_dim = conv_params['decoder_hidden_dim'],
            dropout = conv_params['dropout'],
            learning_rate = conv_params['learning_rate'],
            norm = conv_params['norm'],
            norm_bch = conv_params['norm_bch'],
            add_attn = config['network'].get('add_attn',False),
            n_heads = config['network'].get('n_heads',0),
            clbrt = config['network'].get('clbrt',0)
        )
        
  
            

        log_dir = log_dirs + f'run_{i}/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Build Trainer.

        trainer_fine = pl.Trainer(
                    logger=True,
                    gradient_clip_val = conv_params['gc_fine'],
                    log_every_n_steps = 5,
                    check_val_every_n_epoch=10,
                    default_root_dir= log_dir, 
                    deterministic=True,
                    enable_checkpointing=False,
                    max_epochs=config['epochs'],
                    accelerator="gpu" ,
                    devices=config['loop_dev'],
                    callbacks=[StochasticWeightAveraging(swa_lrs=conv_params['swa']), early_stop_callback],
                    )


        model, exp_info = build_architecture(name=f'exp_transfer_doublenorm_{architecture.__name__}_{"-".join(var_comb["input"])}',
                            architecture = architecture,
                            model_params = model_params,
                            data = data_info,
                            args = {},
                            criterion = criterion,
                            optimizer = optim.Adam,
                            epochs = config['network']['epochs'],
                            class_weight = conv_params.get('cls_wgt',cls_wt),
                            batch_size=conv_params['bs'],
                            n_instances=1,
                            target_dir = log_dir)

        # trainer_fine.logger.log_hyperparams(model_params) #only if trainer logger = False

        trainer_fine.fit(model, Fine_data)

        trainer_fine.save_checkpoint(f"{trainer_fine.logger.log_dir}/best_model.ckpt")
        trainer_fine.test(dataloaders=Fine_data.test_dataloader(), ckpt_path=f"{trainer_fine.logger.log_dir}/best_model.ckpt")

        val_results = trainer_fine.validate(dataloaders=Fine_data.val_dataloader(), ckpt_path=f"{trainer_fine.logger.log_dir}/best_model.ckpt")

        val_ac, val_acc_ts = evaluate_accuracy(model, trainer_fine, Fine_data.val_dataloader(),
                                    config, data_info, var_comb, seasons, 'val')
    
    
        
        trainer_fine.test(dataloaders=Fine_data.test_dataloader(), ckpt_path=f"{trainer_fine.logger.log_dir}/best_model.ckpt")

        fine_acc, fine_acc_ts = evaluate_accuracy(model, trainer_fine, Fine_data.test_dataloader(),config, data_info, var_comb, seasons, 'test')
        
        acc_ts.append(fine_acc_ts)
    
        pred = numpy_predict(model, Fine_data.test_dataloader())
    
        predictions = []
        for i in range(len(pred)):
            predictions.append(pred[i])
        predictions = np.concatenate(predictions)

        result_file = 'predictions.npz'
        np.savez(log_dir + result_file, acc = fine_acc_ts, mean_acc = np.mean(fine_acc_ts), predictions = predictions)

        if not "fine" in setting_training:
            try:
                shutil.rmtree(trainer_fine.logger.log_dir)
            except Exception as e:
                print(f" Error deleting logger path fine tuning: {e}")

    pdb.set_trace()
    accuracy_ts = np.concatenate(acc_ts).reshape(num_mods,6)   

    print(f'Accuracy mean: {accuracy_ts.mean(axis=0)}, var: {accuracy_ts.std(axis=0)}')
    np.savez(f"{log_dirs}/crossval_accuracy_{num_mods}model.npz", 
             mean_acc = accuracy_ts.mean(axis=0), std_acc = accuracy_ts.std(axis=0), var_acc = accuracy_ts.var(axis=0), acc = accuracy_ts)

    config['loaded_pars'] = conv_params
    with open(log_dir + 'config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)