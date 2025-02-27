from datetime import datetime, timedelta

import numpy as np
import xarray
import pdb

import torch
from torch.utils.data import DataLoader
import lightning as pl

# ToDo: Change outline to pass path.

class WeatherDataset(pl.LightningDataModule):
    def __init__(self, dataset_name, data_info, var_comb, seasons=None, return_dates=False):
        self.lag = data_info['config'].get('n_steps_lag')
        self.n_in = data_info['config'].get('n_steps_in')
        self.n_out = data_info['config'].get('n_steps_out')
        self.m_days = data_info['config'].get('mean_days','7')
        self.stack_maps = data_info['config'].get('stack_maps')
        self.regime_path = data_info['config'].get('regime_path','')
        self.data_path = data_info['config'].get('data_path','')
        self.strt = data_info['config'].get('strt','1950')
        self.out_step = data_info['config'].get('out_step', 6)
        self.lon_trafo = data_info['config'].get('lon_trafo', False)
        self.seasons = seasons
        self.return_dates = return_dates
        years = {'WeatherBench': '1979-2018', '20CR': '1836-1980', 'ERA5': '1950-2023'}[dataset_name]
        if 'ERA' in dataset_name: 
            years = f'{self.strt}-2023'
        else:
            years = {'WeatherBench': '1979-2018', '20CR': '1836-1980'}[dataset_name]
        
        key_list = []
        for key in data_info['vars'].keys(): 
            key_list.append(key)
        self.data_info = []
        if 'index' in key_list[0]:
            self.data_info.append(key_list[0])
        elif len(self.data_info) <1:
            self.data_info.append('')

        if 'index' in key_list[1]:
            self.data_info.append(list(key_list)[1])
        elif len(self.data_info) <1:
            self.data_info.append('')

        if 'regimes' in key_list[2]:
            self.data_info.append('')
      
        inputs = {'2d': {}, '1d': []}
        time_steps = set()
        output = None
        for var_name, info in data_info['vars'].items():
            dimension = info['dimension']
            data_type = info['type']

            if var_name == 'nae_regimes':
                resolution = info['resolution']
                pressure_level = info.get('pressure_level', '')
                region = info['region']
                path = f'~/WiOSTNN/Version1/data/{dataset_name}/datasets/{self.regime_path}z_{pressure_level}_{resolution}deg_{years}_{region}_2d_NAEregimes.nc'
                print(path)
            elif 'index' in var_name:
                resolution = info['resolution']
                pressure_level = info.get('pressure_level', '')
                region = info['region']
                path =  f"~/WiOSTNN/Version1/data/{dataset_name}/datasets/{var_name}_{years}_{dataset_name}_s.nc"
            else:
                resolution = info['resolution']
                pressure_level = info.get('pressure_level', '')
                region = info['region']
                path = f'~/WiOSTNN/Version1/data/{dataset_name}/datasets/{self.data_path}{var_name}_{pressure_level}_{resolution}deg_{years}_{region}_{dimension}.nc'


            # check if variable is to be used as input or output
            if var_name in var_comb['input'] + var_comb['output']:
                ds = xarray.open_dataarray(path)

                
                if self.seasons is not None:
                    # if 'index' in var_name and 'mjo' in region:
                    #     ds = ds.sel(time=ds.season.isin(list(np.array(self.seasons)-30)))
                    # else:
                    #     ds = ds.sel(time=ds.season.isin(self.seasons))
                    ds = ds.sel(time=ds.season.isin(self.seasons))
                else:
                    self.seasons=np.unique(ds.season.values)
                try:
                    if 'index' in var_name:
                        ds = ds['u'].squeeze()
                    elif 'index' in var_name and 'MJO' in region:
                        ds = ds.squeeze()
                    else:
                        ds = ds[var_name].squeeze()
                except:
                    ds = ds.squeeze()

                time_steps.add(len(ds.time))
            else:
                continue
            
            # convert categorical variables to one-hot encoding
            if data_type == 'categorical':
                try:
                    categories = ds.data.astype(int)
                except:
                    categories = ds.values.astype(int)

                # add a category dimension 
                if 'index' in var_name and 'mjo' in region:
                    categories = (categories- np.ones(categories.shape)).astype(int)
                    ds.values = categories
                    ds_norm = ds.expand_dims(dim={'mjo_cat': int(np.max(categories)+1)}).transpose('time', 'mjo_cat')
                else:
                    ds_norm = ds.expand_dims(dim={'nae_regime_cat': int(np.max(categories)+1)}).transpose('time', 'nae_regime_cat')
                # convert to one-hot encoding
                ds_norm = ds_norm.copy(data=np.eye(max(categories)+1)[categories])
    
            if data_type == 'index' and not 'mjo' in region:
                # add a category dimension 
                mean = ds.mean(dim=['time'])
                std = ds.std(dim=['time'])
                ds_norm = (ds - mean) / std
            
            # normalize continuous variables if they are used as input
            # then append to inputs dictionary
            if var_name in var_comb['input']:
                if data_type == 'continuous':
                    mean = ds.mean(dim=['time'])
                    std = ds.std(dim=['time'])
                    ds_norm = (ds - mean) / std
                    ds_norm = ds_norm.fillna(0)
                    if self.lon_trafo:
                        #convert from -180,180 to 0,360
                        ds_norm = ds_norm.assign_coords(lon= ds.lon%360)
                        ds_norm = ds_norm.sortby('lon')
                if dimension == '2d':
                    try:
                        inputs['2d'][region].append(ds_norm)
                    except KeyError:
                        inputs['2d'][region] = [ds_norm]
                elif dimension == '1d':

                    inputs['1d'].append(ds_norm) 
            
            if var_name in var_comb['output']:
                    output = ds
            
        self.inputs = inputs
        self.output = output
  
        # compute input and output shapes for model
        if 'nae_regimes' in var_comb['output']:
            output_shape = (self.n_out, 4)
        else:
            output_shape = (self.n_out, *self.output.isel(time=0).values.shape)
        self.shapes = {'input': [], 'output': output_shape}
        if len(self.inputs['2d']) < 1: 
            num_cats = 0 
            for data in self.inputs['1d']:
                try:
                    num_cats += data.shapes[1]
                except:
                    num_cats += 1
            self.shapes['input'].append((self.n_in,num_cats)) 
        else:
            for region, data in self.inputs['2d'].items():
                lat, lon = data[0].isel(time=0).squeeze().values.shape
                if data_info['config']['stack_maps']:
                    self.shapes['input'].append((self.n_in, len(data), lat, lon))
                else:
                    for _ in data:
                        self.shapes['input'].append((self.n_in, 1, lat, lon))

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
        
        season_idx = np.digitize(idx, self.n_samples_per_season_accumulated)    
        season = self.seasons[season_idx]

        idx = idx - self.n_samples_per_season_accumulated[season_idx-1] if season_idx > 0 else idx
        inputs = self.multiple_input(season, season_idx, idx)

        output, output_slice = self.single_outputs(season, season_idx, idx)

        if self.return_dates:
            dates = [datetime.fromisoformat((str(d)[:10])).timetuple().tm_yday for d in output_slice.time.values]
            r = (inputs, output, torch.tensor(dates), output_slice.time.values)
        else:
            r = (inputs, output)

        return r

    def single_outputs(self, season, season_idx, idx):
        """
        Returns the multiple outputs for a given season and season index.
        Parameters:
            season (str): The season for which to retrieve the outputs.
            season_idx (int): The index of the season.
        Returns:
            torch.Tensor: The multiple outputs for the specified season and season index.
        """
        out_idxs = [idx + (self.n_in + self.lag + self.out_step) * self.m_days]
        output_slice = self.output.sel(time=self.output.season == season).isel(time=out_idxs)
        output = torch.tensor(output_slice.values, dtype=torch.long)

        return output, output_slice


    def multiple_input(self, season, season_idx, idx):
        
        inputs = []
        in_idxs = [idx + i*self.m_days for i in range(self.n_in)]

        inputs = self.input_2d(inputs, season, in_idxs)
        inputs = self.input_1d(inputs, season, in_idxs)

        return inputs
    
    def input_2d(self, inputs, season, in_idxs):

        if len(self.inputs['2d']) >= 1:
            for i, (_, data) in enumerate(self.inputs['2d'].items()):
                data = [d.sel(time=d.season == season).isel(time=in_idxs).values for d in data]
                inp = np.stack(data, axis=1)
                inp = self.fix_axes(inp)
                if self.stack_maps:
                    if i == 0:
                        inpts = inp
                    else: 
                        inpts = np.concatenate((inpts,inp), axis=1)
                else: 
                    inpts = inp
            inputs.append(torch.tensor(inpts, dtype=torch.float32))
        else:
            inputs.append([])

        return inputs

    def input_1d(self, inputs, season, in_idxs):

        for nd, d in enumerate(self.inputs['1d']):
            if 'MJO' in self.data_info[nd]:
                if nd == 0:
                    inps = torch.tensor(d.sel(time=d.season == season).isel(time=in_idxs).values.squeeze(), 
                            dtype=torch.float32)
                else:
                    inp = torch.tensor(d.sel(time=d.season == season).isel(time=in_idxs).values.squeeze(), 
                            dtype=torch.float32)
                    inps = torch.cat((inps, inp), dim = 1)
        
            elif 'pv' in self.data_info[nd]: 
                if nd == 0:
                    inps = torch.tensor(d.sel(time=d.season == season).isel(time=in_idxs).values[:,None], 
                            dtype=torch.float32)

                elif nd == 1:
                    inp = torch.tensor(d.sel(time=d.season == season).isel(time=in_idxs).values[:,None], 
                            dtype=torch.float32)
                    inps = torch.cat((inps, inp), dim = 1)

            else:
                if nd == 0:
                    inps = torch.tensor(d.sel(time=d.season == season).isel(time=in_idxs).values,
                            dtype=torch.float32)
                else:
                    inp = torch.tensor(d.sel(time=d.season == season).isel(time=in_idxs).values,
                           dtype=torch.float32)
                    inps = torch.cat((inps, inp), dim = 1)
        
        inputs.append(inps)

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

    
class TransferData(pl.LightningDataModule):

    def __init__(self, dataset_name, var_comb, data = None, batchsize = 32, **params):
        
        super().__init__()
        
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

    def train_dataloader(self):
        self.dataset['train']= WeatherDataset(
                    dataset_name=self.name,
                    data_info=self.data,
                    var_comb=self.var_comb,
                    seasons=self.seasons['train'][self.name]
                )
        return DataLoader(self.dataset['train'], batch_size = self.bs, shuffle=True)
    
    def val_dataloader(self):
        self.dataset['val']= WeatherDataset(
                    dataset_name=self.name,
                    data_info=self.data,
                    var_comb=self.var_comb,
                    seasons=self.seasons['val'][self.name]
                )
        return DataLoader(self.dataset['val'], batch_size = self.bs, shuffle=False)
    
    def test_dataloader(self):
        self.dataset['test'] = WeatherDataset(
                    dataset_name=self.name,
                    data_info=self.data,
                    var_comb=self.var_comb,
                    seasons=self.seasons['test'][self.name]
                )
            
        return DataLoader(self.dataset['test'], batch_size = self.bs, shuffle=False)
    
    def access_dataset(self):
        return self.dataset
