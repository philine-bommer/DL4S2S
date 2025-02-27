import os
import yaml
from argparse import ArgumentParser
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import pdb

import fsspec
import xarray as xr
import numpy as np
from sklearn.decomposition import PCA

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import torch.nn as nn
import lightning
import lightning.pytorch as pl



from aurora import AuroraSmall, Batch, Metadata, Aurora, rollout

# DL packages.
from dataset.dataset_embeddings import ImageDataset, EmbeddingDataset
from dataset.datasets_wrapped import WeatherDataset
from build_model import build_architecture, build_finetune
from utils_data import cls_weights
from utils_evaluation import evaluate_accuracy
from utils import statics_from_config, get_params_from_best_model, get_params_from_model_obj 


activations = {}
def _get_activation(name: str):

    """
    Returns a hook function that stores the activations of a layer during the forward pass.
    Parameters:
    -----------
    name : str
        The name of the layer.
    Returns:
    --------
    Callable[[nn.Module, Any, torch.Tensor], None]
        A hook function that captures and processes the layer’s output.
    """
    def hook(model: nn.Module, input: Any, output: torch.Tensor) -> None:
            activations[name] = output
    return hook


# Load config and settings.
cfd = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open(f'{cfd}/config/aurora_embeddings_config.yaml'), Loader=yaml.FullLoader)

pl.seed_everything(42)

var_comb = config['var_comb']

data_info, _ = statics_from_config(config)
data_info['config']['lon_trafo'] = config['data'].get('lon_trafo',False)

seasons =  {'train':{config['data']['dataset_name2']:list(range(config['data']['fine']['train_start'], config['data']['fine']['train_end']))},
    'val':{config['data']['dataset_name2']:list(range(config['data']['fine']['val_start'], config['data']['fine']['val_end']))},
    'test':{config['data']['dataset_name2']:list(range(config['data']['fine']['test_start'], config['data']['fine']['test_end']))}}


# Create data loader.
params = {'seasons': seasons, 'test_set_name':config['data'][config['name']]['setname']}

# Get the data.
images = {'train': None, 'val': None, 'test': None}
dataset_order = ['train', 'val', 'test']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

root_path = config['root']
data_path = f"{root_path}/Embeddings/Aurora/"
if not os.path.exists(Path(data_path)):
        os.makedirs(Path(data_path))


for keys in dataset_order:

    data = WeatherDataset(dataset_name=config['data']['dataset_name2'],
                        data_info=data_info,
                        var_comb=var_comb,
                        seasons=seasons[keys][config['data']['dataset_name2']],
                        return_dates=True
                    )

    params['return_dates'] = True
    params['lon_trafo'] = config['data'].get('lon_trafo', False)
    images[keys] = ImageDataset(data, config['data']['dataset_name2'], 
                            var_comb, data = None, **params)

    statics = images[keys].statics
    print(f'{keys} done.')


# Import the model and set hooks.
model = Aurora()
model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
model.backbone.register_forward_hook(_get_activation("backbone"))

model.to(device)
# Get the embeddings.
embeddings = {'train': None, 'val': None, 'test': None}

for keys, images in images.items():
    file_emb = f"{data_path}{keys}_embeddings_cleaned.npz"

    if os.path.exists(file_emb):
        surf_vars = images.surf
        atmos_vars = images.atmos
        times = images.times
        idx = images.idx_list
        static_vars_ds = images.statics

        corrs = []
        varc = []
        pca_expl = []
        # colus = {}
        for i in range(len(surf_vars)):
            var_shape = surf_vars[i].shape
            lons = statics.lon.values

            batch = Batch(
                    surf_vars={
                        "2t":surf_vars[i],
                    },
                    static_vars={
                        # The static variables are constant, so we just get them for the first time. They
                        # don't need to be flipped along the latitude dimension, because they are from
                        # ERA5.
                        "slt": torch.from_numpy(static_vars_ds.values[0]),
                    },
                    atmos_vars={
                        "u": atmos_vars[i],
                    },
                    metadata=Metadata(
                        # Flip the latitudes! We need to copy because converting to PyTorch, because the
                        # data must be contiguous.
                        lat=torch.from_numpy(statics.lat.values.copy()),
                        lon=torch.from_numpy(lons),
                        # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
                        # `datetime.datetime`s. Note that this needs to be a tuple of length one:
                        # one value for every batch element.
                        time=(times[i],),
                        atmos_levels=(50,),
                    ),
                )
            model(batch.to(device))
            emb = activations["backbone"].cpu().detach().numpy().squeeze().astype(np.float32)
            pca = PCA(n_components=5)
            pca_full = PCA(n_components=emb.T.shape[0])
            emb_full = pca_full.fit_transform(emb.T).T
            pca_expl.append(pca_full.explained_variance_ratio_)
            del emb_full, pca_full

            emb = pca.fit_transform(emb.T).T
            # embed_shape = emb.shape
            # corr_vec = np.corrcoef(emb)
            # corr_low = corr_vec[np.triu_indices(corr_vec.shape[0], k=1)]
            # rows, cols = np.nonzero(corr_vec > 0.1)
            # r_end = (rows == 0).sum()
            # vec_cols = cols[1:r_end]
            corrs.append(emb)#np.delete(emb,vec_cols,0)
            varc.append(pca.explained_variance_ratio_.sum())
            
            # colus[i] = vec_cols
            print(f'sample {i} with {pca.explained_variance_ratio_.sum()} var done in {keys}.')
            # del emb, corr_vec, corr_low, rows, cols, r_end, vec_cols
           
        embed_shape = emb.shape
        # dt_emb=np.array(corrs)
        # del corrs
        # pdb.set_trace()
        # # Extract the arrays from the dictionary
        # arrays = list(colus.values())

        # # Find common entries across all arrays
        # common_entries = arrays[0]
        # for arr in arrays[1:]:
        #     common_entries = np.intersect1d(common_entries, arr)
        
        # dt_emb_new = np.delete(dt_emb, common_entries, axis=2)
        dt_emb_new = corrs
        varc = np.array(varc)

        # np.savez(file_emb, embeddings=np.array(dt_emb_new), idx=idx)
        np.savez(f"{data_path}{keys}_variance.npz", variance=pca_expl)
        # print("Common entries:", common_entries)
        print("Average variance:", varc.mean())
        del emb, idx, batch, corrs, dt_emb_new#surf_vars, atmos_vars, times, static_vars_ds, lons


del model, images, data, activations
# Save the embeddings.

pdb.set_trace()
var_comb = {"input":["embeddings", "nae_regimes"], "output":["nae_regimes"]}
data_info['config']['embeddings_inputs'] = var_shape
data_info['config']['embeddings'] = embed_shape

dataset_name = config['data']['dataset_name2']
data_info['dataset_name'] = dataset_name
data_info['var_comb'] = var_comb
data_info['seasons'] = seasons
data_info['config']['nae_path'] = f'WiOSTNN/Version1/data/{dataset_name}/datasets/'


with open(f"{data_path}" + 'config.yaml', 'w') as outfile:
    yaml.dump(data_info, outfile, default_flow_style=False)
# for sets, embeds in embeddings.items():
for keys in dataset_order:
    file_emb = f"{data_path}{keys}_embeddings_cleaned.npz"
    embeds = np.load(file_emb)
    
    dataset = EmbeddingDataset(data = embeds['embeddings'].astype(np.float32),
                    dataset_name=config['data']['dataset_name2'],
                    data_info=data_info,
                    var_comb=var_comb,
                    seasons=seasons[keys][config['data']['dataset_name2']],
                    return_dates=False
                )

    torch.save(dataset, f'{data_path}/{keys}_dataset.pt')
    del dataset, embeds
        
    
for keys in dataset_order:
    file_emb = f"{data_path}{keys}_embeddings.npz"
    embeds = np.load(file_emb)
    
    dataset_withTime = EmbeddingDataset(data = embeds['embeddings'].astype(np.float32),
                    dataset_name=config['data']['dataset_name2'],
                    data_info=data_info,
                    var_comb=var_comb,
                    seasons=seasons[keys][config['data']['dataset_name2']],
                    return_dates=True
                )

    torch.save(dataset_withTime, f'{data_path}/{keys}_dataset_withTime.pt')
    del dataset_withTime

# file_z = f"{data_path}embeddings.npz"
# if not os.path.exists(file_z):

#     for keys in dataset_order:

#         data = WeatherDataset(dataset_name=config['data']['dataset_name2'],
#                             data_info=data_info,
#                             var_comb=var_comb,
#                             seasons=seasons[keys][config['data']['dataset_name2']],
#                             return_dates=True
#                         )

#         params['return_dates'] = True
#         params['lon_trafo'] = config['data'].get('lon_trafo', False)
#         images[keys] = ImageDataset(data, config['data']['dataset_name2'], 
#                                 var_comb, data = None, **params)

#         statics = images[keys].statics
#         print(f'{keys} done.')


#     # Import the model and set hooks.
#     model = Aurora()
#     model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
#     model.backbone.register_forward_hook(_get_activation("backbone"))

#     model.to(device)
#     # Get the embeddings.
#     embeddings = {'train': None, 'val': None, 'test': None}

#     for keys, images in images.items():
#         surf_vars = images.surf
#         atmos_vars = images.atmos
#         times = images.times
#         idx = images.idx_list
#         static_vars_ds = images.statics
#         embed = []
#         for i in range(len(surf_vars)):
#             var_shape = surf_vars[i].shape
#             lons = statics.lon.values

#             batch = Batch(
#                     surf_vars={
#                         "2t":surf_vars[i],
#                     },
#                     static_vars={
#                         # The static variables are constant, so we just get them for the first time. They
#                         # don't need to be flipped along the latitude dimension, because they are from
#                         # ERA5.
#                         "slt": torch.from_numpy(static_vars_ds.values[0]),
#                     },
#                     atmos_vars={
#                         "u": atmos_vars[i],
#                     },
#                     metadata=Metadata(
#                         # Flip the latitudes! We need to copy because converting to PyTorch, because the
#                         # data must be contiguous.
#                         lat=torch.from_numpy(statics.lat.values.copy()),
#                         lon=torch.from_numpy(lons),
#                         # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
#                         # `datetime.datetime`s. Note that this needs to be a tuple of length one:
#                         # one value for every batch element.
#                         time=(times[i],),
#                         atmos_levels=(50,),
#                     ),
#                 )
#             model(batch.to(device))
#             embed.append(activations["backbone"].cpu().detach().numpy().squeeze().astype(np.float32))
#         embed_shape = embed[0].shape
#         embeddings[keys] = {'embeddings': np.array(embed), 'idx': idx}

#     del model, batch, embed, images, data, surf_vars, atmos_vars, times, idx, static_vars_ds, activations,lons
    
#     try: 
#         np.savez(file_z, train=embeddings['train']['embeddings'], train_idx=embeddings['train']['idx'], 
#                  val=embeddings['val']['embeddings'], val_idx=embeddings['val']['idx'], 
#                  test=embeddings['test']['embeddings'], test_idx=embeddings['test']['idx'])
#     except Exception as e:
#         pass

# else:
    
#     embed = np.load(file_z)
#     embeddings = {'train': None, 'val': None, 'test': None}
#     for key in embed.files:
#         if '_idx' in key:
#             embeddings[key.replace('idx_','')]['idx'] = embed[key].astype(int)
#         elif isinstance(embed[key], dict):
#             embeddings[key]['embeddings'] = embed[key]['embeddings'].astype(np.float32)
#             embeddings[key]['idx'] = embed[key]['idx'].astype(int)
#         else:
#             embeddings[key]['embeddings'] = embed[key].astype(np.float32)
#         del embed

# # Save the embeddings.

# # pdb.set_trace()
# var_comb = {"input":["embeddings", "nae_regimes"], "output":["nae_regimes"]}
# data_info['config']['embeddings_inputs'] = var_shape
# data_info['config']['embeddings'] = embed_shape

# for sets, embeds in embeddings.items():
    
#     dataset = EmbeddingDataset(data = embeds['embeddings'].astype(np.float32),
#                     dataset_name=config['data']['dataset_name2'],
#                     data_info=data_info,
#                     var_comb=var_comb,
#                     seasons=seasons[sets][config['data']['dataset_name2']],
#                     return_dates=False
#                 )

#     torch.save(dataset, f'{data_path}/{sets}_dataset.pt')
#     del dataset
        
    

# for sets, embeds in embeddings.items():
    
#     dataset_withTime = EmbeddingDataset(data = embeds['embeddings'].astype(np.float32),
#                     dataset_name=config['data']['dataset_name2'],
#                     data_info=data_info,
#                     var_comb=var_comb,
#                     seasons=seasons[sets][config['data']['dataset_name2']],
#                     return_dates=True
#                 )

#     torch.save(dataset_withTime, f'{data_path}/{sets}_dataset_withTime.pt')
#     del dataset_withTime

