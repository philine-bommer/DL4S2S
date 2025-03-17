from datetime import datetime, timedelta

import numpy as np
import xarray
import pdb
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import lightning as pl
import xarray as xr

class ImageDataset(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling image datasets.
    Args:
        dataset: The dataset object.
        var_comb: The variable combination.
        data: The weather variable.
        **params: Additional parameters.
    Attributes:
        name (str): The name of the dataset.
        seasons (list): The list of seasons.
        return_dates (bool): Whether to return dates.
        combine_test (bool): Whether to combine test data.
        var_comb (str): The variable combination.
        data (Any): The weather variable.
        dataset: The dataset object.
    Methods:
        get_images: Retrieves images from the dataset.
        _prepare: Prepare a variable.
    """
    def __init__(self, dataset, dataset_name, var_comb, data = None, **params):

        super().__init__()
        
        self.name = dataset_name
        self.seasons = params.get('seasons',None) 
        self.return_dates = params.get('return_dates', False)
        self.combine_test = params.get('combine_test',False)
        self.var_comb = var_comb
        self.data_dir = params.get('data_dir', '.../Data/')
        self.return_dates = params.get('return_dates', False)
        self.t_len = params.get('t_len', 6)
        self.lon_360 = params.get('lon_tarfo', True)
        # if data is None: 
        #     raise Exception("Weather variable need to be specified")
        # else:
        #     self.data = data

        self.dataset= dataset
        self.get_images()
        download_path = params.get('path',Path(f"{self.data_dir}/ERA5/datasets/"))

        self.statics = xr.open_dataarray(download_path / "static_regrid_northern_hemi.nc", engine="netcdf4")
        if self.lon_360:
            self.statics = self.statics.assign_coords(lon= self.statics.lon%360)
            self.statics = self.statics.sortby('lon')

    def get_images(self):
        """
        Retrieves images from the dataset.

        Returns:
            surf (list): List of preprocessed surface images.
            atmos (list): List of preprocessed atmosphere images.
            idx_list (list): List of corresponding indices.
        """

        self.surf = []
        self.atmos = []
        self.idx_list = []
        self.times = []
        t_step = np.arange(1,6,2)
        for idx in range(self.dataset.__len__()):
            if self.return_dates:
                inputs, _, _, times = self.dataset.__getitem__(idx)
            else:
                inputs, _ = self.dataset.__getitem__(idx)
            images, _ = inputs
            
            surface =  self.fix_axes(images[:,0,:,:])
            atmosphere = self.fix_axes(images[:,1,:,:])[:,None,:,:]
            for t in t_step:
                if self.return_dates:
                    time_stp = self.to_datetime(times[t].astype("datetime64[s]"))
                    self.times.append(time_stp)
                
                self.surf.append(self._prepare(surface,t))
                self.atmos.append(self._prepare(atmosphere,t))
                self.idx_list.append(idx)
        
    
    def _prepare(self,x: np.ndarray,i) -> torch.Tensor:
        """Prepare a variable.

        This does the following things:
        * Select time indices `i` and `i - 1`.
        * Insert an empty batch dimension with `[None]`.
        * Flip along the latitude axis to ensure that the latitudes are decreasing.
        * Copy the data, because the data must be contiguous when converting to PyTorch.
        * Convert to PyTorch.
        """
        return x[[i - 1, i]][None][..., :, :]
    
    def fix_axes(self, x: np.ndarray) -> np.ndarray:
        """Fix the axes of an array.

        This does the following things:
        * Flip along the latitude axis to ensure that the latitudes are decreasing.
        * Remove the last latitude, because it is the same as the first.
        """
        for a,ax in enumerate(x.shape):
            if ax == self.t_len:
                x = x.swapaxes(0,a)
        return x


    def to_datetime(self, date):
        """
        Converts a numpy datetime64 object to a python datetime object 
        Input:
        date - a np.datetime64 object
        Output:
        DATE - a python datetime object
        """
        timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                    / np.timedelta64(1, 's'))
        return datetime.utcfromtimestamp(timestamp)


class EmbeddingDataset(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling image datasets.
    Args:
        dataset: The dataset object.
        var_comb: The variable combination.
        data: The weather variable.
        **params: Additional parameters.
    Attributes:
        name (str): The name of the dataset.
        seasons (list): The list of seasons.
        return_dates (bool): Whether to return dates.
        combine_test (bool): Whether to combine test data.
        var_comb (str): The variable combination.
        data (Any): The weather variable.
        dataset: The dataset object.
    Methods:
        get_images: Retrieves images from the dataset.
        _prepare: Prepare a variable.
    """
    def __init__(self, data, dataset_name, data_info, var_comb, seasons=None, return_dates=False):
        self.lag = data_info['config'].get('n_steps_lag')
        self.n_in = data_info['config'].get('n_steps_in')
        self.n_out = data_info['config'].get('n_steps_out')
        self.m_days = data_info['config'].get('mean_days','7')
        self.stack_maps = data_info['config'].get('stack_maps')
        self.regime_path = data_info['config'].get('regime_path','')
        self.data_path = data_info['config'].get('data_path','')
        self.strt = data_info['config'].get('strt','1950')
        self.nae_path = data_info['config'].get('nae_path',f'.../Data/{dataset_name}/datasets/')
        self.multiple = data_info['config'].get('multiple', True)
        self.seasons = seasons
        self.embed = data_info['config'].get('embed',2)
        self.return_dates = return_dates
        if 'ERA' in dataset_name: 
            years = f'{self.strt}-2023'
        else:
            years = {'20CRv3': '1836-1980'}[dataset_name]
      

        inputs = {}
        time_steps = set()
        output = None

        input_shape = {}
        self.shapes = {'input': {}, 'output': None}
        for var_name in var_comb['input'] + var_comb['output']:

            if var_name == 'nae_regimes':
     
                resolution = data_info['vars'][var_name]['resolution']
                pressure_level = data_info['vars'][var_name].get('pressure_level', '')
                region = data_info['vars'][var_name]['region']
                data_type = data_info['vars'][var_name]['type']
                path = f'{self.nae_path}{self.regime_path}z_{pressure_level}_{resolution}deg_{years}_{region}_2d_NAEregimes.nc'

                ds = xarray.open_dataarray(path)

                
                if self.seasons is not None:
                    ds = ds.sel(time=ds.season.isin(self.seasons))
                else:
                    self.seasons=np.unique(ds.season.values)
                try:
                    ds = ds[var_name].squeeze()
                except:
                    ds = ds.squeeze()
                
                time_steps.add(len(ds.time))

                
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
                    num_cats = max(categories)+1
                
                # normalize continuous variables if they are used as input
                # then append to inputs dictionary
                if not var_name in list(inputs.keys()) and var_name in var_comb['input']:
                    inputs[var_name]= ds_norm 
                
                if var_name in var_comb['output']:
                    output = ds
                
                if 'nae_regimes' in var_comb['output']:
                    output_shape = (self.n_out, num_cats)
                if 'nae_regimes' in var_comb['input']:
                    input_shape[var_name]=(self.n_in,num_cats)
            else:
                if var_name in var_comb['input']:
                    
                    emb_steps = int(self.n_in/data_info['config'][f'{var_name}_inputs'][1])
                    emb_h, emb_v = data_info['config'][var_name]
                    inputs[var_name]= data#.reshape(int(data.shape[0]/emb_steps),emb_steps,emb_h,emb_v)
                    input_shape[var_name] = (emb_steps,emb_h,emb_v)

        self.shapes['output'] = output_shape   
        self.shapes['input'] = input_shape
            
        self.inputs = inputs
        self.output = output
        del ds, inputs, output
  

        # compute number of samples
        assert len(time_steps) == 1, 'All variables must have the same number of time steps'
        self.n_samples_per_season = {}
        for season in self.seasons:
            s = self.output.time.sel(time=self.output.season == season)
            self.n_samples_per_season[season] = max(len(s) - (self.n_in + self.lag + self.n_out) * 7, 0)

        self.n_samples_per_season_accumulated = [sum(list(self.n_samples_per_season.values())[:i]) for i in range(1,len(self.seasons)+1)]

    def multiple_outputs(self, season, season_idx, idx):
        """
        Returns the multiple outputs for a given season and season index.
        Parameters:
            season (str): The season for which to retrieve the outputs.
            season_idx (int): The index of the season.
        Returns:
            torch.Tensor: The multiple outputs for the specified season and season index.
        """
        out_idxs = [idx + (self.n_in + self.lag + i) * self.m_days for i in range(self.n_out)]
        output_slice = self.output.sel(time=self.output.season == season).isel(time=out_idxs)
        output = torch.tensor(output_slice.values, dtype=torch.long)

        return output, output_slice

    def multiple_input(self, season, season_idx, idx):
        
        inputs = []
        for var_name, data in self.inputs.items():

            in_len = self.shapes['input'][var_name][0]

            if 'nae_regimes' in var_name:
                in_idxs = [idx + i*self.m_days for i in range(self.n_in)]
                data = data.sel(time=data.season == season).isel(time=in_idxs).values
                data = torch.tensor(data, dtype=torch.long)
                inputs.append(data)
            else:
                in_len = self.shapes['input'][var_name][0]#int(self.n_in/self.embed)
                in_idxs = [idx + i*self.m_days for i in range(in_len)]
                inputs.append(data[in_idxs])

        return inputs
    

    def __len__(self):
        return sum(list(self.n_samples_per_season.values()))

    def __getitem__(self, idx):

        season_idx = np.digitize(idx, self.n_samples_per_season_accumulated)    
        season = self.seasons[season_idx]
        idx = idx - self.n_samples_per_season_accumulated[season_idx-1] if season_idx > 0 else idx
        inputs = self.multiple_input(season, season_idx, idx)

        output, output_slice = self.multiple_outputs(season, season_idx, idx)

        if self.return_dates:
            dates = [datetime.fromisoformat((str(d)[:10])).timetuple().tm_yday for d in output_slice.time.values]
            r = (inputs, output, torch.tensor(dates), output_slice.time.values)
        else:
            r = (inputs, output)

        return r

class CustomEmbeddingDataset(pl.LightningDataModule):
    def __init__(self, images, regimes, outputs):
        # or use the RobertaTokenizer from `transformers` directly.

        self.images = images
        self.regimes = regimes
        self.out_regimes = outputs
        
    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        inputs = (self.images[i], self.regimes[i])
        outputs = self.out_regimes[i]

        return (inputs, outputs)

class BaseDataset(pl.LightningDataModule):
    def __init__(self, inputs, outputs):
        
        self.input = inputs
        self.labels = outputs
     
        
    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        inputs = self.input[i]
        outputs = self.labels[i]

        return (inputs, outputs)