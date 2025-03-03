import os
import yaml
from argparse import ArgumentParser


import torch
from torch import dropout, nn, utils
import torch.optim as optim
import numpy as np

# DL packages.
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping


import deepS2S.model.ViT as vit
from deepS2S.dataset.datasets_static import SingleData
import deepS2S.model.mae as Mae

if __name__ == '__main__':

    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--notification_email", type=str, default="pbommer@atb-potsdam.de")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default=1)
    parser.add_argument("--config", type=str, default='ViT')
    args = parser.parse_args()

    cfile = args.config

    # Load config and settings.
    exd = os.path.dirname(os.path.abspath(__file__))
    cfd = exd.parent.absolute()
    config = yaml.load(open(f'{cfd}/config/config_pretrain{cfile}.yaml'), Loader=yaml.FullLoader)

    config['net_root'] = str(cfd.parent.absolute()) + f'/Data/Network/'
    config['root'] = str(cfd.parent.absolute()) + f'/Data/Network/Sweeps/'

    pl.seed_everything(42)

    var_comb = config['var_comb']

    data_info = {
    'vars': {
        var_comb[1]['input'][0]: {
            'dimension': '2d',
            'region': 'mjo',
            'resolution': config['data']['resolution'],
            'type': 'continuous'
        },
        var_comb[0]['input'][0]: {
            'dimension': '2d',
            'region': 'spv',
            'resolution': config['data']['resolution'],
            'pressure_level': config['data']['pressure'],
            'type': 'continuous'
        },
        var_comb[0]['output'][0]: {
            'dimension': '1d',
            'type': 'categorical',
            'resolution': config['data']['resolution'],
            'pressure_level': config['data']['nae'],
            'region': 'northern_hemi'
        }
    },
    'config':{
        'regime_path':config['data'].get('regime_path',''),
        'data_path':config['data'].get('data_path',''),
        'strt':config.get('strt',''),
        'n_steps_in': config['data']['n_steps_in'],
        'n_steps_lag': config['data']['lag'],
        'n_steps_out': config['data']['n_steps_out'],
        'stack_maps': False,  # if False, each map is treated as a separate input. Not implemented yet.
        'test_set_name': config['data'][config['name']]['setname'],
        'test_set_seasons': list(range(config['data'][config['name']]['test_start'],config['data'][config['name']]['test_end']))
    }
    }
    seasons =  {'train':{config['data']['dataset_name1']:list(range(config['data']['pre']['train_start'], config['data']['pre']['train_end'])),
                        config['data']['dataset_name2']:list(range(config['data']['fine']['train_start'], config['data']['fine']['train_end']))},
               'val':{config['data']['dataset_name1']:list(range(config['data']['pre']['val_start'], config['data']['pre']['val_end'])),
                      config['data']['dataset_name2']:list(range(config['data']['fine']['val_start'], config['data']['fine']['val_end']))},
               'test':{config['data']['dataset_name1']:list(range(config['data']['pre']['test_start'], config['data']['pre']['test_end'])),
                       config['data']['dataset_name2']:list(range(config['data']['fine']['test_start'], config['data']['fine']['test_end']))},}



    log_dir = config['net_root'] + 'MAE/' 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    strt_yr = config.get('strt','')
    trial_num = config.get('version', '')
    norm_opt = config.get('norm_opt','')
    log_dir = log_dir + f'version_{strt_yr}{trial_num}_{norm_opt}/individual_static/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create data loader.
    for combinations in var_comb:
        params = {'seasons': seasons, 'test_set_name':config['data'][config['name']]['setname']}
        Pre_data = SingleData(config['data']['dataset_name1'], config['data']['dataset_name2'],
                                    combinations, data_info, config['data']['bs'], **params)
        Pre_data.train_dataloader()
        Pre_data.val_dataloader()
        test_loader = Pre_data.test_dataloader()
        
        data = Pre_data.access_dataset()
  
        frame_size = [int(x) for x in data['val'][0][0][0].shape][-2:]
        input_dim = [int(x) for x in data['val'][0][0][0].shape][0]
        # remember for reproducibility and analysis
        data_info['var_comb'] = var_comb

        # Set up models args.
        encoder = vit.ViT(image_size = tuple(frame_size),
                        patch_size = tuple(config['vit']['patch_size']),
                        num_classes = config['vit']['num_classes'],
                        dim = config['vit']['dim'],
                        depth = config['vit']['depth'],
                        heads = config['vit']['heads'],
                        mlp_dim = config['vit']['mlp_dim'],
                        dropout = config['vit']['dropout'],
                        emb_dropout = config['vit']['emb_dropout'],
                        channels = config['vit']['channels'])

        mae = Mae.MAE(
            encoder = encoder,
            masking_ratio = config['mae']['masking_ratio'],   # the paper recommended 75% masked patches
            decoder_dim = config['mae']['decoder_dim'],      # paper showed good results with just 512
            decoder_depth = config['mae']['decoder_depth']       # anywhere from 1 to 8
        )

        for x,y in test_loader:
            x_enc = mae.get_image_embedding(x)
            enc_shape = x_enc.shape[1:]
    


        early_stop_callback = EarlyStopping(monitor="val_loss", 
                                            min_delta=0.00, patience=3, verbose=True, mode="min")

            # Build Trainer.
        trainer_pre = pl.Trainer(accelerator=args.accelerator,
                                devices=[0,1],
                                check_val_every_n_epoch=10,  
                                max_epochs=config['epochs'], 
                                default_root_dir=log_dir, 
                                deterministic=True,
                                callbacks=[early_stop_callback],)


        if config['devices'] >=1:
            device = torch.device("cuda")

        # Build model
        model_params = dict(
            input_dim=input_dim,
            in_time_lag=config['data']['n_steps_in'],
            out_time_lag=config['data']['n_steps_out'],
            out_dim=data['val'][0][1].shape[-1],
            frame_size=frame_size,
            output_probabilities=True,
            add_attn = config['network'].get('add_attn',False),
            n_heads = config['network'].get('n_heads',0),
            clbrt = config['network'].get('clbrt',0)
        )
        
        # Run Training.
        var_name = combinations['input'][0]
        trainer_pre.fit(mae, Pre_data)

        trainer_pre.save_checkpoint(f"{trainer_pre.logger.log_dir}/best_model_{var_name}.ckpt")
        trainer_pre.test(dataloaders=test_loader, ckpt_path=f"{trainer_pre.logger.log_dir}/best_model_{var_name}.ckpt")


        config['loaded_pars'] = model_params
        config['enc_shape'] = list(enc_shape)
        
        with open(log_dir + f'config_{var_name}.yaml', 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)