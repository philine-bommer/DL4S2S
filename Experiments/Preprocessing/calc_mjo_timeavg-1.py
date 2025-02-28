#!/usr/bin/env python3


import argparse
import xarray as xr
import numpy as np
import pdb

def get_args():
    """  Get command line arguments. """
    parser = argparse.ArgumentParser(
        description="""Calculate running mean of RMM indices produced by Extravert and rederive 
        other properties (i.e. amplitude, phase, etc). """
    )
    parser.add_argument(
        "infile",
        type=str,
        help="Path to netcdf file produced by Extravert containing daily MJO RMM indices.",
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="Path for output netcdf file.",
    )
    parser.add_argument(
        "ndays",
        type=int,
        help="Width of running mean (in days).",
    )
    parser.add_argument(
        "--timedim",
        type=str,
        default="time",
        help="Name of time dimension.",
    )

    args = parser.parse_args()

    return args


def calc_phase_number(phase):
    """Return DataArray of MJO phase number determined from phase angle. Note that
    phase number is calculated for all dates, regardless of MJO amplitude."""
    phase_number = phase.copy(deep=True).rename("phase_number")
    phase_number[:] = np.nan
    phase_number[(phase >= -180) & (phase < -135)] = 1
    phase_number[(phase >= -135) & (phase < -90)] = 2
    phase_number[(phase >= -90) & (phase < -45)] = 3
    phase_number[(phase >= -45) & (phase < 0)] = 4
    phase_number[(phase >= 0) & (phase < 45)] = 5
    phase_number[(phase >= 45) & (phase < 90)] = 6
    phase_number[(phase >= 90) & (phase < 135)] = 7
    phase_number[(phase >= 135) & (phase <= 180)] = 8
    return phase_number

def main(args):
    # Read data
    ds = xr.open_dataset(args.infile, engine ="netcdf4")

    time_coords = ds.time.values[int(args.ndays/2):-int(args.ndays/2)]
    # Calculate rolling average
    avg = ds.rolling(**{args.timedim:args.ndays, "center":True}).mean(skipna=True)
   
    avg = avg.sel(time=slice(avg.time.values[int(args.ndays/2)], avg.time.values[-int(args.ndays/2)-1])).assign_coords(time=time_coords)

    
    # Recalculate properties time-averaged RMM1 and RMM2
    phase = (np.arctan2(avg.rmm2, avg.rmm1) * 180 / np.pi).dropna("time")
    amplitude = (np.sqrt(avg.rmm1 * avg.rmm1 + avg.rmm2 * avg.rmm2)).dropna("time")
    phase_number = calc_phase_number(phase).dropna("time")

    # Insert values back into avg dataset
    avg = avg.assign(phase = phase)
    avg = avg.assign(amplitude = amplitude)
    avg = avg.assign(phase_number = phase_number)

    
    avg.to_netcdf(args.outfile)


if __name__ == "__main__":
    main(get_args())
