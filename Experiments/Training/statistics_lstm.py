import os
import yaml
from argparse import ArgumentParser
import shutil
from pathlib import Path

import torch
import pdb
from torch import dropout, nn, utils
import torch.optim as optim
import numpy as np

# DL packages.
import lightning
import lightning.pytorch as pl
from lightning.pytorch.callbacks import StochasticWeightAveraging, EarlyStopping


from deepS2S.model import ViTLSTM
from deepS2S.model.loss import FocalLossAdaptive
from deepS2S.dataset.datasets_regimes import TransferData
from deepS2S.utils.utils_build import build_architecture, build_encoder
from deepS2S.utils.utils_data import cls_weights
from deepS2S.utils.utils_evaluation import evaluate_accuracy, numpy_predict
from deepS2S.utils.utils import statics_from_config, get_params_from_best_model

if __name__ == '__main__':

    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--notification_email", type=str, default="pbommer@atb-potsdam.de")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default=1)
    parser.add_argument("--network", type=str, default='ViT_LSTM')
    parser.add_argument("--config", type=str, default='')
    parser.add_argument("--ntrials", default=100)
    args = parser.parse_args()

    cfile = args.config

    # Load config and settings.
    exd = os.path.dirname(os.path.abspath(__file__))
    cfd = Path(exd).parent.absolute()
    config = yaml.load(open(f'{cfd}/config/config_lstm.yaml'), Loader=yaml.FullLoader)

    config['net_root'] = str(cfd.parent.absolute()) + f'/Data/Network/'
    config['root'] = str(cfd.parent.absolute()) + f'/Data/Network/Sweeps/'
    config['data_root'] = str(cfd.parent.absolute()) + f'/Data'

    num_mods = args.ntrials

    seeds = np.arange(1,num_mods +1, dtype = int)

    # Set up models args.
    ntype = args.network
    # Set up models args
    net = 'ViT_LSTM'
    config['network']['mode'] = 'base'
    setting_training = "fine"


    config['root'] = config['root'] + f"{ntype}/"
    config['arch'] = ''
    architecture = ViTLSTM.ViT_LSTM
    conv_params = get_params_from_best_model(config, 'ViT_LSTM')

    config['arch'] = ''
    var_comb = config['var_comb']

    data_info, seasons = statics_from_config(config)


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
    data_info['var_comb'] = var_comb

    log_dir = config['net_root'] + 'Statistics/LSTM/'         
    print(f'log: {log_dir}')


    strt_yr = config.get('strt','')
    trial_num = config.get('version', '')
    norm_opt = config.get('norm_opt','')
    name_var = config.get('tropics','')
    vit_dir = f'version_{strt_yr}{trial_num}_{norm_opt}{name_var}'
    log_dirs = log_dir + f'version_{strt_yr}{trial_num}_{norm_opt}{name_var}/'
    if not os.path.exists(log_dirs):
        os.makedirs(log_dirs)

    early_stop_callback = EarlyStopping(monitor="train_acc", 
                                        min_delta=0.00, patience=3, verbose=True, mode="max")

    if config['devices']:
        device = torch.device("cuda")

    if 'calibrated'in norm_opt:
        criterion = FocalLossAdaptive(gamma = conv_params['gamma'].get('val',3), 
                                                          numerical = conv_params['gamma'].get('numeric',False), 
                                                          device = device)
    
    # # Build model
    # # Build encoder.
    strt_yr = config.get('strt','')
    trial_num = config.get('version', '')
    norm_opt = config.get('norm_opt','')
    name_var = config.get('tropics','')
    Mae_tropics, Mae_u = build_encoder(config)
    enc_path = config['net_root'] + f'MAE/version_{strt_yr}{trial_num}_{norm_opt}/individual_static/'
    config_enc = yaml.load(open(f'{enc_path}config_u.yaml'), Loader=yaml.FullLoader)


    config_enc_u = yaml.load(open(f'{enc_path}config_u.yaml'), Loader=yaml.FullLoader)

    model_params = dict(
        encoder_u = Mae_u.encoder,
        encoder_sst = Mae_tropics.encoder,
        enc_out_shape =  [1,config_enc_u['vit']['dim']],
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
        add_attn = config['network'].get('add_attn', False),
        n_heads = config['network'].get('n_heads',6),
        clbrt = config['network'].get('clbrt',0),
        mode = config['network'].get('mode', 'run'))
    
    acc_ts = []
    for i in seeds:
        pl.seed_everything(i)

        log_dir = log_dirs + f'run_{i}/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        model, exp_info = build_architecture(name=f'LSTM_{"-".join(var_comb["input"])}',
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


            # Run Training.
        trainer_fine = pl.Trainer(accelerator=args.accelerator,
                                        gradient_clip_val = conv_params['gc_fine'],
                                        devices = config['devices'], 
                                        log_every_n_steps = 10,
                                        max_epochs=config['epochs'], 
                                        check_val_every_n_epoch=5,
                                        default_root_dir=log_dir, 
                                        deterministic=True,
                                        callbacks=[StochasticWeightAveraging(swa_lrs=conv_params['swa']),
                                                early_stop_callback],)
            
        trainer_fine.fit(model, Fine_data)
                
        trainer_fine.save_checkpoint(f"{trainer_fine.logger.log_dir}/best_model.ckpt")

        fine_acc, fine_acc_ts = evaluate_accuracy(model, trainer_fine, Fine_data.test_dataloader(),config, data_info, var_comb, seasons, 'test')

        acc_ts.append(fine_acc_ts)
        
        pred = numpy_predict(model, Fine_data.test_dataloader())
       
        predictions = []
        for i in range(len(pred)):
            predictions.append(pred[i])
        predictions = np.concatenate(predictions)

        result_file = 'predictions.npz'
        np.savez(log_dir + result_file, acc = fine_acc_ts, mean_acc = np.mean(fine_acc_ts), predictions = predictions)

        
        config['loaded_pars'] = conv_params
        with open(log_dir + 'config.yaml', 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)


    accuracy_ts = np.concatenate(acc_ts).reshape(num_mods,config['data']['n_steps_out'])

    print(f'Accuracy mean: {accuracy_ts.mean(axis=0)}, var: {accuracy_ts.std(axis=0)}')
    np.savez(f"{log_dirs}/lstm_accuracy_{num_mods}model.npz", 
             mean_acc = accuracy_ts.mean(axis=0), std_acc = accuracy_ts.std(axis=0), var_acc = accuracy_ts.var(axis=0), acc = accuracy_ts)