# %%
import os
import yaml
from argparse import ArgumentParser
from pathlib import Path
import warnings
import copy
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

import pdb

import numpy as np
import xarray as xr
import xesmf as xe


from climatology import *




# set variable downloaded from the 20th century Reanalysis Data
parser = ArgumentParser()

parser.add_argument("--vrbl", type=str, default='pressure')
parser.add_argument("--clima", type=str, default='month')
args = parser.parse_args()

vrbl = args.vrbl
clima = args.clima
config = yaml.load(open(f"/mnt/beegfs/home/bommer1/WiOSTNN/Experiments/preprocessing/config/{vrbl}_data.yml"), Loader=yaml.FullLoader)


# User Configuration : Change these values to generate different datasets
var_name = config.get('var_name','z') # available: 't2m', 'tp', 'z'
pressure_level = config.get('pressure_level',500) # in hPa, ignored for variables without levels. available: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1,000
region = config.get('region','northern_hemi') # available: 'global', 'europe', 'northern_hemi' 
resolution = config.get('resolution',1.40625) 
start_year, end_year = config.get('start_year', 1980), config.get('end_year', 2023) 
reduce_to_single_dim = config.get('reduce_to_single_dim', False) # if True, the dataset will be reduced to a single dimension (time) by aggregating over the other dimensions
months_to_keep = config.get('months_to_keep', [11, 12, 1, 2, 3])
start_season, end_season = config.get('start_season','11-15'), config.get('end_season','03-31')
start_offset, end_offset  = config.get('start_offset','11-01'), config.get('end_offset','04-15')

region = config.get('region','northern_hemi')
if region == 'global':
    region_coords = None
else:
    region_coords = {'lat': slice(config[region][0], config[region][1])}

aggregate_func = config.get('aggregate_func', 'mean') # available: 'mean', 'sum', 'std', 'var', 'min', 'max'
var_name_map = config.get('var_map','geopotential')#{'sst': 'air', 'z': 'geopotential', 'u': 'uwnd','olr':'ulwrf'}
var_full = config.get('var_full','z')#{'sst': 't2m', 'z': 'z', 'u': 'u','olr':'ulwrf'}

full_data_path = Path(f'./raw/{var_name}/{var_name}_{pressure_level}_full.nc')
full_data_p = f'./raw/{var_name}/{var_name}_{pressure_level}_full.nc'

if not os.path.exists(Path(f'./raw/{var_name}/')):
        os.makedirs(Path(f'./raw/{var_name}/'))
if not full_data_path.exists(): 
    if var_name == 'z':
        in_file = f'./raw/{var_name_map}_{var_name}{pressure_level}.nc'
        ds = xr.open_dataset(in_file)
        # change units to watts
        ds = ds/9.81
    elif var_name == 'u':
        in_file = f'./raw/{var_name_map}_{pressure_level}.nc'
        ds = xr.open_dataset(in_file)
    elif var_name == 'olr':
        ds = xr.open_mfdataset(f'./raw/{var_name}/{var_name}_{pressure_level}*.nc', combine='by_coords', engine ="netcdf4")

        # change units to watts
        ds = - ds/3600

    yrs = np.arange(start_year, end_year+1)

    for i in range(len(yrs)-1):
        year = yrs[i]
        ds_sel = ds.sel({'valid_time':slice(np.datetime64(f'{year}-{start_offset}'),np.datetime64(f'{year+1}-{end_offset}'))})
        if 'ttr' in var_name_map:
            ds_sel = ds_sel.astype('float32')
        else:
            de_sel = ds_sel.astype('float16')
        ds_sel = preprocess_data(ds_sel, 
                    region_coords, 
                    aggregate_func)
        if 'ttr' in var_name_map:
            test = ds_sel[var_name_map]
        else:
            test = ds_sel[var_name]
        assert np.isnan(test.values).any() == 0, 'There are NaN values in the dataset after preprocess.'
        del test
        if i ==0:
            dst = ds_sel
        else:
            dst = xr.concat((dst,ds_sel), dim ='time')
    dst.to_netcdf(full_data_p)
    del ds_sel, dst

ds = xr.open_dataarray(full_data_p)    
    # ds = ds.resample(time='1D').mean('time')

grid_path = Path(f'../WeatherBench/{resolution}deg_grid.nc')

if grid_path.exists():
    grids = xr.open_dataset(grid_path)
    grids['lat'] = -grids['lat']
    grids =  grids.sel(**region_coords)

days = config.get('days', 7)

try: 
    ds = xr.open_mfdataset(f'./raw/{var_name}/{var_name}_{pressure_level}_grid_r-mean{days}_*.nc', combine='by_coords', engine ="netcdf4").__xarray_dataarray_variable__
except:
    yrs = np.arange(start_year, end_year+1)

    for i in range(len(yrs)-1):
        year = yrs[i]
        ds_sel = ds.sel({'time':slice(np.datetime64(f'{year}-{start_offset}'),np.datetime64(f'{year+1}-{end_offset}'))})

        ds_sel = ds_sel.astype('float32')

        # apply rolling mean
        ds_sel = apply_rolling_mean(ds_sel, days, year, start_season, year +1, end_season, months_to_keep)


        if grid_path.exists():
            # Mirror latitude values around 0
            regridder = xe.Regridder(ds_sel, grids, "bilinear")
            try:
                if 'ttr' in var_name_map:
                    ds_sel = regridder(ds_sel[var_name_map])
                else:
                    ds_sel = regridder(ds_sel[var_name])
            except:
                ds_sel = regridder(ds_sel)
        else:
            print('No grid found. Skipping regridding.')

        assert np.isnan(ds_sel.values).any() == 0, 'There are NaN values in the dataset after regridding.'
        print(f'Regrid and rolling mean {year} - {year+1} done.')
        # if i ==0:
        #     dst = ds_sel
        # else:
        #     dst = xr.concat((dst,ds_sel), dim ='time')
        sel_name = f'./raw/{var_name}/{var_name}_{pressure_level}_grid_r-mean{days}_{year}-{year+1}.nc'
        ds_sel.to_netcdf(sel_name, engine ="netcdf4")

    ds = xr.open_mfdataset(f'./raw/{var_name}/{var_name}_{pressure_level}_grid_r-mean{days}_*.nc', combine='by_coords', engine ="netcdf4")

# calculate climatology and anomalies
climatology_folder = Path('./climatology/monthly')
climatology_folder.mkdir(exist_ok=True)

climatology_path = climatology_folder / Path(f'{var_name}_{resolution}deg_{pressure_level}_climatology.nc') 
climatology_p = f'./climatology/{var_name}_{resolution}deg_{pressure_level}_{clima}ly_climatology_' 

anom_path = Path(f'./raw/{var_name}/{clima}')
anom_path.mkdir(exist_ok=True)
anomalies = f'./raw/{var_name}/{clima}/{var_name}_{pressure_level}_anom.nc'
if not Path(anomalies).exists():
# if 'uwnd' in var_name_map:
    anom = climatology_and_anomalies_perLat(ds, config.get('years',30), climatology_p, clima)
    print(f'Anomaly calculation done.')
# else:
#     anom = climatology_and_anomalies(ds, config.get('years',30), climatology_p, clima)

    anom.to_netcdf(anomalies, engine ="netcdf4")

ds = xr.open_dataarray(anomalies, engine ="netcdf4")
yrs = np.arange(start_year, end_year+1)
# seas = []
for i in range(len(yrs)-1):
    year = yrs[i]
    ds_sel = ds.sel({'time':slice(np.datetime64(f'{year}-{start_season}'),np.datetime64(f'{year+1}-{end_season}'))}).resample(time='1D').mean('time')
    len_time = len(ds_sel.time.values)
    seas_coord = np.repeat(i+1,len_time)
    ds_sel = ds_sel.assign_coords(season=('time', seas_coord))
    assert np.isnan(ds_sel.values).any() == 0, 'There are NaN values in the dataset in selected season.'
    
    if i > 0:
        dst = xr.concat((dst,ds_sel), dim ='time')
        seas.append(len_time)
    if i ==0:
        seas = [len_time,len_time]
        dst = ds_sel
        del ds_sel
print(f'Season assignment done.')    
# seas.append(len_time)
ds = dst
del ds_sel, dst


if clima == 'dai':
    dataset_dir = Path(f'./datasets/{clima}')
else:
    dataset_dir = Path(f'./datasets')
dataset_dir.mkdir(exist_ok=True)

file_name = dataset_dir / Path(f'{var_name}_{pressure_level}_{resolution}deg_{start_year}-{end_year}_{region}_{"1d" if reduce_to_single_dim else "2d"}.nc')
# ds = ds.assign_coords(season=('time', season_coords))

ds.to_netcdf(file_name, engine ="netcdf4")
print(xr.load_dataarray(file_name, engine ="netcdf4"))

if vrbl in ['olr','pv']:
    small_reg = config.get('small_reg','')
    try:
        ds =ds.sel({'latitude': slice(config['small'][0], config['small'][1])})
    except:
        ds =ds.sel({'lat': slice(config['small'][0], config['small'][1])})
    file_name = dataset_dir / Path(f'{var_name}_{pressure_level}_{resolution}deg_{start_year}-{end_year}_{small_reg}_{"1d" if reduce_to_single_dim else "2d"}.nc')
    ds.to_netcdf(file_name, engine ="netcdf4")  
