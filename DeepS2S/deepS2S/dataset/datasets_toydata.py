from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import pdb
import copy

import torch
from torch.utils.data import DataLoader
import lightning as pl



class ToyDataset(pl.LightningDataModule):
    def __init__(self, data_info, ensembles, return_class=False):

        self.return_class = return_class
        self.lag = data_info['config'].get('n_steps_lag')
        self.n_in = data_info['config'].get('n_steps_in')
        self.n_out = data_info['config'].get('n_steps_out')
        self.m_days = data_info['config'].get('mean_days',7)
        self.ensembles = ensembles
        
        self.stack_maps = data_info['config'].get('stack_maps')
    
        

        inputs = {'2d': [], '1d': []}
        input_shape = {'2d': None, '1d': None}
        self.shapes = {}
        output = None

        data = xr.open_dataset(f'/mnt/beegfs/home/bommer1/WiOSTNN/Version1/data/ToyData/datasets/ToyData.nc')
        data = data.sel({'ensembles': self.ensembles})
        data = data.stack(time=("samples", "ensembles"))
        ratios = copy.deepcopy(data.ratios)
        # data = data.swap_dims({'time': 'lat'})
        

        for dimension, _ in inputs.items():
            if dimension == '2d':
                inputs[dimension] = data.__xarray_dataarray_variable__
                input_shape[dimension] = inputs[dimension].shape

            else:
                output = data.ratios
                output_shape = (np.prod(output.shape), 1)
                
                classes = copy.deepcopy(data.ratios)
                classes[classes > 0] = 1
                classes[classes < 0] = 0
                classe = classes.values.astype('int')
                class_vals = np.eye(max(classe)+1)[classe]
                classes = classes.expand_dims(dim={'nae_regime_cat': int(np.max(classe)+1)}).transpose('time', 'nae_regime_cat')
                # convert to one-hot encoding
                ds_norm = classes.copy(data=class_vals)
                inputs[dimension].append(ds_norm)
                input_shape[dimension] = (np.prod(classes.shape), 2)
  
            
        self.shapes['input'] = input_shape
        self.shapes['output'] = output_shape    
        self.inputs = inputs
        self.output = output
        
        self.n_samples_per_season = {}
        for season in self.ensembles:
            s = self.output.samples.sel(ensembles=season)
            self.n_samples_per_season[season] = max(len(s) - (self.n_in + self.lag + self.n_out) * self.m_days, 0)
    
        
        
        
    def __len__(self):
        return sum(list(self.n_samples_per_season.values()))
     
    def __getitem__(self, idx):
    
        inputs = self.multiple_input(idx)

        output, output_slice = self.multiple_outputs(idx)

        r = (inputs, output)

        return r
    
    def multiple_outputs(self, idx):
        """
        Returns the multiple outputs for a given season and season index.
        Parameters:
            season (str): The season for which to retrieve the outputs.
            season_idx (int): The index of the season.
        Returns:
            torch.Tensor: The multiple outputs for the specified season and season index.
        """
        out_idxs = [idx + (self.n_in + self.lag + i) * self.m_days for i in range(self.n_out)]
        output_slice = self.output.isel(time=out_idxs)
        output = torch.tensor(output_slice.values, dtype=torch.long)

        return output, output_slice


    def multiple_input(self, idx):
        
        inputs = []
        in_idxs = [idx + i*self.m_days for i in range(self.n_in)]

        inputs = self.input_2d(inputs, in_idxs)
        inputs = self.input_1d(inputs, in_idxs)

        return inputs
    
    def input_2d(self, inputs, in_idxs):

        data = self.inputs['2d']
        if self.stack_maps:
            for a,ax in enumerate(data.shape):
                if ax == 2:
                    data = data.swapaxes(1,a)
        else: 
            data = data
        inpts = data.isel(time=in_idxs).values.swapaxes(1,-1).swapaxes(-1,-2).swapaxes(0,1)
        inputs.append(torch.tensor(inpts, dtype=torch.float32))

        return inputs

    def input_1d(self, inputs, in_idxs):

        data = self.inputs['1d'][0]
        inpts = data.isel(time=in_idxs).values
        inputs.append(torch.tensor(inpts, dtype=torch.float32))
    
        return inputs
    
    def fix_axes(self, x: np.ndarray) -> np.ndarray:
        """Fix the axes of an array.

        This does the following things:
        * Flip along the latitude axis to ensure that the latitudes are decreasing.
        * Remove the last latitude, because it is the same as the first.
        """
        for a,ax in enumerate(x.shape):
            if ax == self.n_in:
                x = x.swapaxes(0,a)
        return x

    
class ToyDataLoader(pl.LightningDataModule):

    def __init__(self, data, batchsize = 32, **params):
        
        super().__init__()
        
        self.data =data
        self.bs = batchsize
        self.dataset = {'train': [], 'val': [],'test': []}
        self.ensembles = params.get('ensembles',None) 
        self.return_class = params.get('return_class', False)

    def train_dataloader(self):
        self.dataset['train']= ToyDataset(
                    data_info=self.data,
                    ensembles=self.ensembles['train'],
                    return_class=self.return_class
                )
        return DataLoader(self.dataset['train'], batch_size = self.bs, shuffle=True)
    
    def val_dataloader(self):
        self.dataset['val']= ToyDataset(
                    data_info=self.data,
                    ensembles=self.ensembles['val'],
                    return_class=self.return_class
                )
        return DataLoader(self.dataset['val'], batch_size = self.bs, shuffle=False)
    
    def test_dataloader(self):
        self.dataset['test'] = ToyDataset(
                    data_info=self.data,
                    ensembles=self.ensembles['test'],
                    return_class=self.return_class
                )
            
        return DataLoader(self.dataset['test'], batch_size = self.bs, shuffle=False)
    
    def access_dataset(self):
        return self.dataset


class ToySingleData(pl.LightningDataModule):
    def __init__(self, data_info, ensembles, return_class=False):

        self.return_class = return_class
        self.lag = data_info['config'].get('n_steps_lag')
        self.n_in = data_info['config'].get('n_steps_in')
        self.n_out = data_info['config'].get('n_steps_out')
        self.m_days = data_info['config'].get('mean_days',7)
        self.ensembles = ensembles
        self.var_name = data_info['config'].get('var_name')
        
        self.stack_maps = data_info['config'].get('stack_maps')
        

        inputs = {'2d': [], '1d': []}
        input_shape = {'2d': None, '1d': None}
        self.shapes = {}
        output = None

        data = xr.open_dataset(f'/mnt/beegfs/home/bommer1/WiOSTNN/Version1/data/ToyData/datasets/ToyData.nc')
        data = data.sel({'ensembles': self.ensembles})
        data = data.stack(time=("samples", "ensembles"))
        ratios = copy.deepcopy(data.ratios)
        # data = data.swap_dims({'time': 'lat'})

        for dimension, _ in inputs.items():
            if dimension == '2d':
                inputs[dimension] = data.__xarray_dataarray_variable__.sel({'types': self.var_name})
                input_shape[dimension] = inputs[dimension].shape

            else:
                inputs[dimension].append(ratios)
                if self.return_class:
                    classes = copy.deepcopy(data.ratios)
                    classes[classes > 0] = 1
                    classes[classes < 0] = 0
                    output = classes
                    output_shape = (np.prod(classes.shape), 2)
                else:
                    output = data.ratios
                    output_shape = (np.prod(output.shape), 1)
                input_shape[dimension] = (np.prod(output.shape), 1)
            
        self.shapes['input'] = input_shape
        self.shapes['output'] = output_shape    
        self.inputs = inputs
        self.output = output
        
        
  
        
    def __len__(self):
        return len(self.inputs['2d'].time.values)
     
    def __getitem__(self, idx):
    
        inputs = self.multiple_input(idx)

        output, output_slice = self.multiple_outputs(idx)

        r = (inputs, output)

        return r
    
    def multiple_outputs(self, idx):
        """
        Returns the multiple outputs for a given season and season index.
        Parameters:
            season (str): The season for which to retrieve the outputs.
            season_idx (int): The index of the season.
        Returns:
            torch.Tensor: The multiple outputs for the specified season and season index.
        """
        out_idxs = idx#[idx + (self.n_in + self.lag + i) * self.m_days for i in range(self.n_out)]
        output_slice = self.output.isel(time=out_idxs)
        output = torch.tensor(output_slice.values[None,...], dtype=torch.long)

        return output, output_slice


    def multiple_input(self, idx):
        
        inputs = []
        in_idxs = idx#[idx + i*self.m_days for i in range(self.n_in)]

        inputs = self.input_2d(inputs, in_idxs)

        return inputs
    
    def input_2d(self, inputs, in_idxs):

        data = self.inputs['2d']
        inpts = data.isel(time=in_idxs).values.swapaxes(0,-1).swapaxes(-1,-2)[None,...]
        inputs = torch.tensor(inpts, dtype=torch.float32)

        return inputs

    def input_1d(self, inputs, in_idxs):

        data = self.inputs['1d']
        inpts = data.isel(time=in_idxs).values
        inputs.append(torch.tensor(inpts, dtype=torch.float32))
    
        return inputs
    

    
class ToySingleLoader(pl.LightningDataModule):

    def __init__(self, data, batchsize = 32, **params):
        
        super().__init__()
        
        self.data =data
        self.bs = batchsize
        self.dataset = {'train': [], 'val': [],'test': []}
        self.ensembles = params.get('ensembles',None) 
        self.return_class = params.get('return_class', False)

    def train_dataloader(self):
        self.dataset['train']= ToySingleData(
                    data_info=self.data,
                    ensembles=self.ensembles['train'],
                    return_class=self.return_class
                )
        return DataLoader(self.dataset['train'], batch_size = self.bs, shuffle=True)
    
    def val_dataloader(self):
        self.dataset['val']= ToySingleData(
                    data_info=self.data,
                    ensembles=self.ensembles['val'],
                    return_class=self.return_class
                )
        return DataLoader(self.dataset['val'], batch_size = self.bs, shuffle=False)
    
    def test_dataloader(self):
        self.dataset['test'] = ToySingleData(
                    data_info=self.data,
                    ensembles=self.ensembles['test'],
                    return_class=self.return_class
                )
            
        return DataLoader(self.dataset['test'], batch_size = self.bs, shuffle=False)
    
    def access_dataset(self):
        return self.dataset


