from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import copy
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import pdb



def climatology_and_anomalies(data: xr.DataArray,
                              years: float,
                              path: str,
                              avg: str = 'month'
                              ) -> xr.DataArray:

        
        data = data.resample(time='1D').mean('time')
        anomalies = copy.deepcopy(data)
        last = data.isel(time=-1)
        first = data.isel(time=0)
        yrs = np.arange(first.time.dt.year.values, last.time.dt.year.values+1)

        beg = '01-01'
        end = '12-31'

        for i in range(len(yrs)//years +1):
            
            if ((i+1)*years) > len(yrs):
                start_dt = np.datetime64(f'{yrs[i*years]}-{beg}')
                end_dt = last.time.values
            elif i == 0:
                start_dt = first.time.values
                end_dt = np.datetime64(f'{yrs[(i+1)*years-1]}-{end}') 
            else: 
                start_dt = np.datetime64(f'{yrs[i*years]}-{beg}')
                end_dt = np.datetime64(f'{yrs[(i+1)*years-1]}-{end}') 
            
            dst = data.sel({'time':slice(start_dt,end_dt)}).astype('float32')
            if i == 0:
                # for first iteration we consider same period for climatology and anomalies
                if 'dai' == avg:
                    clim =  dst.groupby('time.dayofyear').mean('time')
                    anomalies.loc[{'time':slice(start_dt,end_dt)}] = dst.groupby('time.dayofyear') - clim
                else:
                    clim =  dst.groupby('time.month').mean('time')
                    anomalies.loc[{'time':slice(start_dt,end_dt)}] = dst.groupby('time.month') - clim
            else:
                # for the rest of the iterations we consider the previous period for anomalies and the current period for climatology
                if 'dai' == avg:
                    anomalies.loc[{'time':slice(start_dt,end_dt)}] = dst.groupby('time.dayofyear') - clim
                    clim =  dst.groupby('time.dayofyear').mean('time')
                else:
                    anomalies.loc[{'time':slice(start_dt,end_dt)}] = dst.groupby('time.month') - clim
                    clim =  dst.groupby('time.month').mean('time')


            if ((i+1)*years) > len(yrs):
                clim.to_netcdf(path + f'{yrs[i*years]}_{last.time.dt.year.values}.nc', engine ="netcdf4")
            else:
                clim.to_netcdf(path + f'{yrs[i*years]}_{yrs[(i+1)*years-1]}.nc', engine ="netcdf4")

        return anomalies

def longitude_trafo(ds: xr.Dataset) -> xr.Dataset:
    """
    Function to transform the longitude coordinates of a dataset to -180 to 180 degrees.
    """
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.roll(longitude=int(len(ds['longitude']) / 2), roll_coords=True)

    return ds

def preprocess_data(data: xr.Dataset, 
                    region_coords, 
                    aggregate_func) -> xr.Dataset:
    
    data = data.sel(**{'latitude': region_coords['lat']})

    # change longitude from 0-360 to -180-180 for extraction of europe
    if not data.longitude.min().values < 0:
        data = longitude_trafo(data)

    try:
        data = data.rename({'valid_time':'time'})           
    except:
        pass
    # Get daily data
    if aggregate_func == 'mean':
        data = data.resample(time='1D').mean('time')
    
    try:
        data = data.drop('time_bnds')
    except ValueError:
        pass

    try:
        data = data.drop('pressure_level')
        data = data.mean(dim = 'expver')
    except:
        pass
    return data

def apply_rolling_mean(ds: xr.Dataset, 
                       days: int, 
                       start_year: int, 
                       start_season: str, 
                       end_year: int, 
                       end_season: str, 
                       months_to_keep: list) -> xr.Dataset:
    """ Apply rolling mean to dataset
    """

    # apply rolling mean
    time_len = len(ds.time.values)
    time_coords = ds.time.values[:-days + 1]
    ds = ds.rolling(time=days).mean('time')
    ds = ds[{'time':slice((days-1),time_len)}].assign_coords(time=time_coords).dropna("time")
    ds = ds.sel({'time':slice(np.datetime64(f'{start_year}-{start_season}'),np.datetime64(f'{end_year}-{end_season}'))})
    ds = ds.sel(time=np.isin(ds.time.dt.month, months_to_keep))


    return ds

def process_latitude(anomalies, lats, data, start_dates, end_dates, avg, dim_l):
    
    anom = anomalies.loc[{dim_l: lats}]
    ds = data.sel({dim_l: lats}).astype('float32')

    clim_list = []

    for i, (start_dt, end_dt) in enumerate(zip(start_dates, end_dates)):
        dst = ds.sel({'time': slice(start_dt, end_dt)})
        if i == 0:
            # For first iteration we consider same period for climatology and anomalies
            if 'dai' == avg:
                clim_l = dst.groupby('time.dayofyear').mean('time')
                anom.loc[{'time': slice(start_dt, end_dt)}] = dst.groupby('time.dayofyear') - clim_l
            else:
                clim_l = dst.groupby('time.month').mean('time')
                anom.loc[{'time': slice(start_dt, end_dt)}] = dst.groupby('time.month') - clim_l
        else:
            # For the rest of the iterations we consider the previous period for anomalies and the current period for climatology
            if 'dai' == avg:
                anom.loc[{'time': slice(start_dt, end_dt)}] = dst.groupby('time.dayofyear') - clim_l
                clim_l = dst.groupby('time.dayofyear').mean('time')
            else:
                anom.loc[{'time': slice(start_dt, end_dt)}] = dst.groupby('time.month') - clim_l
                clim_l = dst.groupby('time.month').mean('time')

        clim_list.append(clim_l) 


    return anom, clim_list

def climatology_and_anomalies_perLat(data: xr.DataArray, years: float, path: str, avg: str = 'month') -> xr.DataArray:
    anomalies = copy.deepcopy(data)
    last = data.isel(time=-1)
    first = data.isel(time=0)
    yrs = np.arange(first.time.dt.year.values, last.time.dt.year.values + 1)

    beg = '01-01'
    end = '12-31'

    try:
       latitude = data['latitude'].values
       dim_l = 'latitude'
    except:
        latitude = data['lat'].values
        dim_l = 'lat'

    # Precompute the start and end dates for each period
    start_dates = []
    end_dates = []
    for i in range(len(yrs) // years + 1):
        if ((i + 1) * years) > len(yrs):
            start_dates.append(np.datetime64(f'{yrs[i * years]}-{beg}'))
            end_dates.append(last.time.values)
        elif i == 0:
            start_dates.append(first.time.values)
            end_dates.append(np.datetime64(f'{yrs[(i + 1) * years - 1]}-{end}'))
        else:
            start_dates.append(np.datetime64(f'{yrs[i * years]}-{beg}'))
            end_dates.append(np.datetime64(f'{yrs[(i + 1) * years - 1]}-{end}'))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_latitude, anomalies, lats, data, start_dates, end_dates, avg, dim_l) for l, lats in enumerate(latitude)]
        results = [future.result() for future in futures]

    # Combine results
    anoms, clims_list = zip(*results)
    clim_list = combine_arrays(clims_list, latitude)
    anomalies = xr.concat(anoms, dim= dim_l)

    # Save the climatology to a NetCDF file
    for i, (start_dt, end_dt) in enumerate(zip(start_dates, end_dates)):
        clim = clim_list[i]
        if ((i + 1) * years) > len(yrs):
            clim.to_netcdf(path + f'{yrs[i * years]}_{last.time.dt.year.values}.nc', engine="netcdf4")
        else:
            clim.to_netcdf(path + f'{yrs[i * years]}_{yrs[(i + 1) * years - 1]}.nc', engine="netcdf4")

    return anomalies

def combine_arrays(data: list, dim: np.array,)-> list:

    data_dict = {str(k): [] for k in range(len(data[0]))}
    for i, ds in enumerate(data):
        for t, dst in enumerate(ds):
            if i == 0:
                data_dict[f'{t}']= dst
            else:
                data_dict[f'{t}'] = xr.concat((data_dict[f'{t}'],dst), dim = 'latitude')
                # data_dict[f'{t}']['latitude'] = data[0][0].assign_coords(latitude = dim)


    return list(data_dict.values())

# def climatology_and_anomalies_perLat(data: xr.DataArray,
#                               years: float,
#                               path: str,
#                               avg: str = 'month'
#                               ) -> xr.DataArray:

        

#         anomalies = copy.deepcopy(data)
#         last = data.isel(time=-1)
#         first = data.isel(time=0)
#         yrs = np.arange(first.time.dt.year.values, last.time.dt.year.values+1)

#         beg = '01-01'
#         end = '12-31'

#         for i in range(len(yrs)//years +1):
#             if ((i+1)*years) > len(yrs):
#                 start_dt = np.datetime64(f'{yrs[i*years]}-{beg}')
#                 end_dt = last.time.values
#             elif i == 0:
#                 start_dt = first.time.values
#                 end_dt = np.datetime64(f'{yrs[(i+1)*years-1]}-{end}') 
#             else: 
#                 start_dt = np.datetime64(f'{yrs[i*years]}-{beg}')
#                 end_dt = np.datetime64(f'{yrs[(i+1)*years-1]}-{end}') 
#             for l, lats in enumerate(data['latitude'].values):
                
#                 dst = data.sel({'time':slice(start_dt,end_dt),'latitude': lats}).astype('float32')
#                 if i == 0:
#                     # for first iteration we consider same period for climatology and anomalies
#                     if 'dai' == avg:
#                         clim_l =  dst.groupby('time.dayofyear').mean('time')
#                         anomalies.loc[{'time':slice(start_dt,end_dt),'latitude': lats}] = dst.groupby('time.dayofyear') - clim_l
#                     else:
#                         clim_l=  dst.groupby('time.month').mean('time')
#                         anomalies.loc[{'time':slice(start_dt,end_dt),'latitude': lats}] = dst.groupby('time.month') - clim_l
#                 else:
#                     # for the rest of the iterations we consider the previous period for anomalies and the current period for climatology
#                     if 'dai' == avg:
#                         anomalies.loc[{'time':slice(start_dt,end_dt),'latitude': lats}] = dst.groupby('time.dayofyear') - clim_prev.loc[{'latitude': lats}]
#                         clim_l =  dst.groupby('time.dayofyear').mean('time')
#                     else:
#                         anomalies.loc[{'time':slice(start_dt,end_dt),'latitude': lats}] = dst.groupby('time.month') - clim_prev.loc[{'latitude': lats}]
#                         clim_l =  dst.groupby('time.month').mean('time')
#                 if l == 0:
#                     clim = clim_l
#                 else: 
#                     clim = xr.concat((clim,clim_l), dim ='latitude')

#             clim_prev = clim
#             if ((i+1)*years) > len(yrs):
#                 clim.to_netcdf(path + f'{yrs[i*years]}_{last.time.dt.year.values}.nc', engine ="netcdf4")
#             else:
#                 clim.to_netcdf(path + f'{yrs[i*years]}_{yrs[(i+1)*years-1]}.nc', engine ="netcdf4")
#             del clim

#         return anomalies
