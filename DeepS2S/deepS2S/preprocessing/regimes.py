import numpy as np
import xarray as xr




def get_monthly_probabilistic_regimes(ds, regimes):
    """
    Compute the climatology of probabilistic regimes on a monthly basis.

    Parameters
    ----------
    ds : xr.DataArray
        The data array containing the regime information.
    regimes : list
        List of strings giving the names of the regimes.

    Returns
    -------
    ds_cl : xr.DataArray
        The data array containing the monthly climatology of the regimes.
    """
    days_dt = ds.time.dt.month
    day_dt = np.unique(days_dt.values)
    num_cl = np.max(np.unique(ds)) + 1
    ds_clim = np.zeros((day_dt.shape[0], num_cl))

    # Create a dictionary to store the data for each month
    ds_day = {str(d): [] for d in day_dt}

    # Iterate over the data and store the regime values for each month
    for i in range(ds.shape[0]):
        ds_day[str(days_dt.values[i])].append(ds.values[i])

    # Compute the climatology for each month
    for j, (key, val) in enumerate(ds_day.items()):
        unique, cnts = np.unique(val, return_counts=True)
        counts = np.zeros((num_cl,))
        counts[unique] = cnts

        # Normalize the counts to obtain probabilities
        ds_clim[j, :] = counts / counts.sum()

    # Create the data array for the climatology
    ds_cl = xr.DataArray(
        data=ds_clim,
        dims=["month", "regimes"],
        coords=dict(
            month=day_dt,
            regimes=regimes,
        ),
        attrs=dict(
            description="Climatology of NAE regimes",
            units="probability",
        ),
    )

    return ds_cl

def get_general_probabilistic_regimes(ds,regimes):
    """
    Compute the climatology of probabilistic regimes over the entire year.

    Parameters
    ----------
    ds : xr.DataArray
        The data array containing the regime information.
    regimes : list
        List of strings giving the names of the regimes.

    Returns
    -------
    ds_cl : xr.DataArray
        The data array containing the climatology of the regimes over the entire year.
    """
    days_dt = ds.time.dt.dayofyear
    day_dt = np.unique(days_dt.values)
    unique, cnts = np.unique(ds.values, return_counts=True)
    num_cl = np.max(unique)+1
    ds_clim = np.zeros((day_dt.shape[0], num_cl))
    counts = np.zeros((num_cl,))
    counts[unique] = cnts/cnts.sum()

    for i in range(ds_clim.shape[0]):
 
        ds_clim[i, :] = counts

    ds_cl  = xr.DataArray(
    data=ds_clim,
    dims=["dayoftheyear","regimes"],
    coords=dict(
        dayoftheyear = day_dt,
        regimes = regimes,
    ),
    attrs=dict(
        description="Climatology of NAE regimes",
        units="probability",
    ),)
       
    return ds_cl

def get_probabilistic_regimes(ds):
    """
    Compute the daily climatology of probabilistic regimes.

    Parameters
    ----------
    ds : xr.DataArray
        The data array containing the regime information.

    Returns
    -------
    ds_cl : xr.DataArray
        The data array containing the daily climatology of the regimes.
        The dimensions are 'dayoftheyear' and 'regimes', where the
        probabilities of each regime are calculated for each day of the year.
    """

    days_dt = ds.time.dt.dayofyear
    day_dt = np.unique(days_dt.values)
    num_cl = np.max(np.unique(ds))+1
    ds_clim = np.zeros((day_dt.shape[0], num_cl))

    ds_day = {str(d): [] for d in day_dt}

    for i in range(ds.shape[0]):
        ds_day[str(days_dt.values[i])].append(ds.values[i])

    for j, (key, val) in enumerate(ds_day.items()):
        unique, cnts = np.unique(val, return_counts=True)
        counts = np.zeros((num_cl,))
        counts[unique] = cnts

        ds_clim[j,:] = counts/counts.sum() 

    ds_cl  = xr.DataArray(
    data=ds_clim,
    dims=["dayoftheyear","regimes"],
    coords=dict(
        dayoftheyear = day_dt,
        regimes = ['SB', 'NAO-', 'AR', 'NAO+'],
    ),
    attrs=dict(
        description="Climatology of NAE regimes",
        units="probability",
    ),)
       
    return ds_cl

def get_dominant_regime(ds):
    """
    Return the dominant regime for a given time series of regimes.

    Parameters
    ----------
    ds : xr.DataArray
        The data array containing the regime information.

    Returns
    -------
    xr.DataArray
        The data array containing the dominant regime.
    """
    unique, counts = np.unique(ds.values, return_counts=True)
    return xr.DataArray(unique[np.argmax(counts)])