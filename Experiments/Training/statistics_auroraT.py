import os
import yaml
from argparse import ArgumentParser
from pathlib import Path


import pdb
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np # USE EITHER NUMPY OR TORCH 

from deepS2S.model.temporalViT import TemporalTransformerModel
from deepS2S.dataset.dataset_embeddings import EmbeddingDataset
from deepS2S.utils.utils_train import compute_class_weights, training, accuracy_per_timestep

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--ntrials", default=100)
    args = parser.parse_args()

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_mods = args.ntrials

    seeds = np.arange(1,num_mods +1, dtype = int)



    # Load data.
    exd = os.path.dirname(os.path.abspath(__file__))
    cfd = Path(exd).parent.absolute()
    config = yaml.unsafe_load(open(f'{cfd}/config/config_aurora_T.yaml'))

    root_path = str(cfd.parent.absolute())+'/Data/Embeddings/Aurora'
    data_path = f"{root_path}/"

    config['data_root'] = str(cfd.parent.absolute()) + f'/Data'

    dataset_order = ['train', 'val', 'test']
    seasons = config['seasons']
    dataset_name = config['dataset_name']
    config['config']['nae_path'] = f"{root_path}/" #don't change this
    batch_size =  config['network']['batch_size']

    keys = 'train'
    file_emb = f"{data_path}{keys}_embeddings_cleaned.npz"
    embeds = np.load(file_emb)
    train_dataset = EmbeddingDataset(data = embeds['embeddings'].astype(np.float32),
                        dataset_name=config['dataset_name'],
                        data_info=config,
                        var_comb=config['var_comb'],
                        seasons=seasons[keys][config['dataset_name']],
                        return_dates=False
                    )
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)

    keys = 'val'
    file_emb = f"{data_path}{keys}_embeddings_cleaned.npz"
    embeds = np.load(file_emb)
    val_dataset = EmbeddingDataset(data = embeds['embeddings'].astype(np.float32),
                        dataset_name=config['dataset_name'],
                        data_info=config,
                        var_comb=config['var_comb'],
                        seasons=seasons[keys][config['dataset_name']],
                        return_dates=False
                    )
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True)

    keys = 'test'
    file_emb = f"{data_path}{keys}_embeddings_cleaned.npz"
    embeds = np.load(file_emb)
    test_dataset = EmbeddingDataset(data = embeds['embeddings'].astype(np.float32),
                        dataset_name=config['dataset_name'],
                        data_info=config,
                        var_comb=config['var_comb'],
                        seasons=seasons[keys][config['dataset_name']],
                        return_dates=False
                    )

    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

    keys = 'test'
    file_emb = f"{data_path}{keys}_embeddings_cleaned.npz"
    embeds = np.load(file_emb)
    test_dataset = EmbeddingDataset(data = embeds['embeddings'].astype(np.float32),
                        dataset_name=config['dataset_name'],
                        data_info=config,
                        var_comb=config['var_comb'],
                        seasons=seasons[keys][config['dataset_name']],
                        return_dates=False
                    )
    
    log_dirs = '/mnt/beegfs/home/bommer1/WiOSTNN/Data/Network/Statistics/Aurora/' 
    if not os.path.exists(log_dirs):
        os.makedirs(log_dirs)

    acc_ts = []
    for i in seeds:
        torch.manual_seed(i)
        
        random.seed(0)

        log_dir = log_dirs + f'run_{i}/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Initialize model, criterion, optimizer, scheduler
        model = TemporalTransformerModel().to(device)
        num_classes = 4
        learning_rate = 1e-4
        weight_decay = 1e-5  # Added weight decay
        max_epochs = 500

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Assuming train_loader is defined
        class_weights = compute_class_weights(train_loader, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=25, factor=0.5, verbose=True)
        best_val_loss, best_val_acc, model = training(train_loader, val_loader, model, optimizer, scheduler, criterion, 
                                                      max_epochs, num_classes, device, log_dir)
        
        fine_acc_ts = accuracy_per_timestep(model, test_loader, num_classes, device)
        acc_ts.append(fine_acc_ts)


    accuracy_ts = np.concatenate(acc_ts).reshape(num_mods,6)

    print(f'Accuracy mean: {accuracy_ts.mean(axis=0)}, std: {accuracy_ts.std(axis=0)}')
    np.savez(f"/mnt/beegfs/home/bommer1/WiOSTNN/Data/Results/Statistics/ViT/version_1980_calibrated_olr/AURORA_accuracy_97model.npz", 
             mean_acc = accuracy_ts.mean(axis=0), std_acc = accuracy_ts.std(axis=0), var_acc = accuracy_ts.var(axis=0), acc = accuracy_ts)