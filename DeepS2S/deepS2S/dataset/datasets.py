from datetime import datetime, timedelta

import numpy as np
import xarray
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from torch.utils.data import DataLoader
import lightning as pl
import utils as utils

# ToDo: Change to (batch size, TS, h, w)  and only one input,  and (6,1) output

class WeatherDataset(pl.LightningDataModule):
    def __init__(self, dataset_name, data_info, var_comb, seasons=None, return_dates=False):
        self.lag = data_info['config'].get('n_steps_lag')
        self.n_in = data_info['config'].get('n_steps_in')
        self.n_out = data_info['config'].get('n_steps_out')
        self.stack_maps = data_info['config'].get('stack_maps')
        self.regime_path = data_info['config'].get('regime_path','')
        self.data_path = data_info['config'].get('data_path','')
        self.strt = data_info['config'].get('strt','1950')
        self.seasons = seasons
        self.return_dates = return_dates
        years = {'WeatherBench': '1979-2018', '20CR': '1836-1980', 'ERA5': '1950-2023'}[dataset_name]
        if 'ERA' in dataset_name: 
            years = f'{self.strt}-2023'
        else:
            years = {'WeatherBench': '1979-2018', '20CR': '1836-1980'}[dataset_name]

        inputs = {'2d': {}, '1d': []}
        time_steps = set()
        output = None
        info = data_info["vars"]
        vrbls = var_comb['input']+var_comb['output']

        for var_name in vrbls:
            dimension = info[var_name]['dimension']
            data_type = info[var_name]['type']

            if var_name == 'nae_regimes':
                resolution = info[var_name]['resolution']
                pressure_level = info[var_name].get('pressure_level', '')
                region = info[var_name]['region']
                path = f'~/WiOSTNN/Version1/data/{dataset_name}/datasets/{self.regime_path}z_{pressure_level}_{resolution}deg_{years}_{region}_2d_NAEregimes.nc'
            else:
                resolution = info[var_name]['resolution']
                pressure_level = info[var_name].get('pressure_level', '')
                region = info[var_name]['region']
                path = f'~/WiOSTNN/Version1/data/{dataset_name}/datasets/{self.data_path}{var_name}_{pressure_level}_{resolution}deg_{years}_{region}_{dimension}.nc'


            # check if variable is to be used as input or output
            if var_name in var_comb['input'] + var_comb['output']:
                ds = xarray.open_dataarray(path)
                time_steps.add(len(ds.time))
                if self.seasons is not None:
                    ds = ds.sel(time=ds.season.isin(self.seasons))
                else:
                    self.seasons=np.unique(ds.season.values)
                try:
                    ds = ds[var_name].squeeze()
                except:
                    ds = ds.squeeze()
            else:
                continue
            
            # convert categorical variables to one-hot encoding
            if data_type == 'categorical':
                try:
                    categories = ds.data.astype(int)
                except:
                    categories = ds.values.astype(int)
                # add a category dimension 
                ds_norm = ds.expand_dims(dim={'nae_regime_cat': int(np.max(categories)+1)}).transpose('time', 'nae_regime_cat')
                # convert to one-hot encoding
                ds_norm = ds_norm.copy(data=np.eye(max(categories)+1)[categories])
            
            # normalize continuous variables if they are used as input
            # then append to inputs dictionary
            if var_name in var_comb['input']:
                if data_type == 'continuous':
                    mean = ds.mean(dim=['time'])
                    std = ds.std(dim=['time'])
                    ds_norm = (ds - mean) / std
                    ds_norm = ds_norm.fillna(0)
                if dimension == '2d':
                    inputs = ds_norm

        
            
            if var_name in var_comb['output']:
                    output = ds
            
        self.inputs = inputs
        self.output = output
        
       
        # compute input and output shapes for model
        if 'nae_regimes' in var_comb['output']:
            output_shape = (self.n_out,)
        else:
            output_shape = (self.n_out, *self.output.isel(time=0).values.shape)
        self.shapes = {'input': [], 'output': output_shape}

        data = self.inputs
        lat, lon = data.isel(time=0).squeeze().values.shape
        if data_info['config']['stack_maps']:
            self.shapes['input'].append((self.n_in, len(data), lat, lon))
        else:
            self.shapes['input'].append((self.n_in, lat, lon))
        
        # compute number of samples
        assert len(time_steps) == 1, 'All variables must have the same number of time steps'
        self.n_samples_per_season = {}
        for season in self.seasons:
            s = self.output.time.sel(time=self.output.season == season)
            self.n_samples_per_season[season] = max(len(s) - (self.n_in + self.lag + self.n_out) * 7, 0)

        self.n_samples_per_season_accumulated = [sum(list(self.n_samples_per_season.values())[:i]) for i in range(1,len(self.seasons)+1)]
       
        
    def __len__(self):
        return sum(list(self.n_samples_per_season.values()))
    
    def __getitem__(self, idx):
        inputs = []
        season_idx = np.digitize(idx, self.n_samples_per_season_accumulated)    
        season = self.seasons[season_idx]
        idx = idx - self.n_samples_per_season_accumulated[season_idx-1] if season_idx > 0 else idx
        in_idxs = [idx + i*7 for i in range(self.n_in)]
        out_idxs = [idx + (self.n_in + self.lag + i) * 7 for i in range(self.n_out)]
        
        data = self.inputs
        data = [data.sel(time=data.season == season).isel(time=in_idxs).values]

        inpts = np.squeeze(np.stack(data, axis=1))
        inputs = torch.tensor(inpts, dtype=torch.float32)

        output_slice = self.output.sel(time=self.output.season == season).isel(time=out_idxs)
        output = torch.tensor(output_slice.values, dtype=torch.long)

        if self.return_dates:
            dates = [datetime.fromisoformat((str(d)[:10])).timetuple().tm_yday for d in output_slice.time.values]
            r = (inputs, output, torch.tensor(dates), output_slice.time.values)
        else:
            r = (inputs, output)

        return r

class ExperimentalData(pl.LightningDataModule):

    def __init__(self, dataset_name1, dataset_name2, var_comb, data = None, batchsize = 32, **params):
        
        super().__init__()
        
        self.name1 = dataset_name1
        self.name2 = dataset_name2
        self.bs = batchsize
        self.dataset = {'train': [], 'val': [],'test': []}
        self.seasons = params.get('seasons',None) 
        self.return_dates = params.get('return_dates', False)
        self.combine_test = params.get('combine_test',False)
        self.var_comb = var_comb
        if data is None: 
            raise Exception("Weather variable need to be specified")
        else:
            self.data = data

    def train_dataloader(self):
        img_data = {self.name1 : [], self.name2: []}
        for names in [self.name1, self.name2]:
            img_data[names] = torch.utils.data.ConcatDataset([
                    WeatherDataset(
                        dataset_name=names,
                        data_info=self.data,
                        var_comb=self.var_comb[0],
                        seasons=self.seasons['train'][names]
                    ),
                    WeatherDataset(
                        dataset_name=names,
                        data_info=self.data,
                        var_comb=self.var_comb[1],
                        seasons=self.seasons['train'][names]
                    )])
        
        self.dataset['train']= torch.utils.data.ConcatDataset([
                img_data[self.name1], img_data[self.name2]])
        
        return DataLoader(self.dataset['train'], batch_size = self.bs, shuffle=True)
    
    def val_dataloader(self):
        img_data = {self.name1 : [], self.name2: []}
        for names in [self.name1, self.name2]:
            img_data[names]= torch.utils.data.ConcatDataset([
                    WeatherDataset(
                        dataset_name=names,
                        data_info=self.data,
                        var_comb=self.var_comb[0],
                        seasons=self.seasons['val'][names]
                    ),
                    WeatherDataset(
                        dataset_name=names,
                        data_info=self.data,
                        var_comb=self.var_comb[1],
                        seasons=self.seasons['val'][names]
                    )])
            
        self.dataset['val']= torch.utils.data.ConcatDataset([
                img_data[self.name1], img_data[self.name2]])
        
        return DataLoader(self.dataset['val'], batch_size = self.bs, shuffle=False)
    
    def test_dataloader(self):
 
        self.dataset['test'] = torch.utils.data.ConcatDataset([
                WeatherDataset(
                    dataset_name=self.name2,
                    data_info=self.data,
                    var_comb=self.var_comb[0],
                    seasons=self.seasons['test'][self.name2]
                ),
                WeatherDataset(
                    dataset_name=self.name2,
                    data_info=self.data,
                    var_comb=self.var_comb[1],
                    seasons=self.seasons['test'][self.name2]
                )])
            
        return DataLoader(self.dataset['test'], batch_size = self.bs, shuffle=False)
    
    def access_dataset(self):
        return self.dataset
    
class SingleData(pl.LightningDataModule):

    def __init__(self, dataset_name1, dataset_name2, var_comb, data = None, batchsize = 32, **params):
        
        super().__init__()
        
        self.name1 = dataset_name1
        self.name2 = dataset_name2
        self.bs = batchsize
        self.dataset = {'train': [], 'val': [],'test': []}
        self.seasons = params.get('seasons',None) 
        self.return_dates = params.get('return_dates', False)
        self.combine_test = params.get('combine_test',False)
        self.var_comb = var_comb
        if data is None: 
            raise Exception("Weather variable need to be specified")
        else:
            self.data = data

    def train_dataloader(self):
        img_data = {self.name1 : [], self.name2: []}
        for names in [self.name1, self.name2]:
            img_data[names] = WeatherDataset(
                        dataset_name=names,
                        data_info=self.data,
                        var_comb=self.var_comb,
                        seasons=self.seasons['train'][names]
                    )
        
        self.dataset['train']= torch.utils.data.ConcatDataset([
                img_data[self.name1], img_data[self.name2]])
        
        return DataLoader(self.dataset['train'], batch_size = self.bs, shuffle=True)
    
    def val_dataloader(self):
        img_data = {self.name1 : [], self.name2: []}
        for names in [self.name1, self.name2]:
            img_data[names]= WeatherDataset(
                        dataset_name=names,
                        data_info=self.data,
                        var_comb=self.var_comb,
                        seasons=self.seasons['val'][names])
            
        self.dataset['val']= torch.utils.data.ConcatDataset([
                img_data[self.name1], img_data[self.name2]])
        
        return DataLoader(self.dataset['val'], batch_size = self.bs, shuffle=False)
    
    def test_dataloader(self):
 
        self.dataset['test'] = WeatherDataset(
                    dataset_name=self.name2,
                    data_info=self.data,
                    var_comb=self.var_comb,
                    seasons=self.seasons['test'][self.name2]
                )
            
        return DataLoader(self.dataset['test'], batch_size = self.bs, shuffle=False)
    
    def access_dataset(self):
        return self.dataset
    
class PlainData(torch.utils.data.Dataset):

    def __init__(self, dataset_name, var_comb, data = None, batchsize = 32, **params):
        
        super(PlainData, self).__init__()
        
        self.name = dataset_name
        self.bs = batchsize
        self.dataset = {'train': [], 'val': [],'test': []}
        self.seasons = params.get('seasons',None) 
        self.return_dates = params.get('return_dates', False)
        self.combine_test = params.get('combine_test',False)
        self.var_comb = var_comb
        if data is None: 
            raise Exception("Weather variable need to be specified")
        else:
            self.data = data
    
    def train_data(self):
        data = WeatherDataset(
                    dataset_name=self.name,
                    data_info=self.data,
                    var_comb=self.var_comb,
                    seasons=self.seasons['train'][self.name]
                )
        # pdb.set_trace()
        # inputs = data.inputs['1d']
        # outputs = data.output
        x_train , y_train = utils.individual_timestep(data)
        self.dataset['train'] = [x_train, y_train]
        return x_train , y_train
    
    def val_data(self):
        data = WeatherDataset(
                    dataset_name=self.name,
                    data_info=self.data,
                    var_comb=self.var_comb,
                    seasons=self.seasons['val'][self.name]
                )
        # inputs = data.inputs['1d']
        # outputs = data.output
        # pdb.set_trace()
        x_val , y_val = utils.individual_timestep(data)
        self.dataset['val'] = [x_val, y_val]
        return x_val , y_val
    
    def test_data(self):
        data = WeatherDataset(
                    dataset_name=self.name,
                    data_info=self.data,
                    var_comb=self.var_comb,
                    seasons=self.seasons['test'][self.name]
                )
        # inputs = data.inputs['1d']
        # outputs = data.output
        # pdb.set_trace()
        x_test , y_test = utils.individual_timestep(data)
        self.dataset['test'] = [x_test , y_test]
        return x_test , y_test
    

    def access_dataset(self):
        return self.dataset