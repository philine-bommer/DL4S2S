from ftplib import FTP
from argparse import ArgumentParser
from pathlib import Path
import xarray as xr

ftp = FTP('ftp2.psl.noaa.gov')
ftp.login('anonymous', 'paulboehnke95@gmail.com')

# set variable downloaded from the 20th century Reanalysis Data
parser = ArgumentParser()

parser.add_argument("--vrbl", type=str, default='pressure')
args = parser.parse_args()

vrbl = args.vrbl


# Download files 1980 - 2015.
if vrbl == 'pressure':
    var = 'hgt'
    directory = '/Datasets/20thC_ReanV3/Dailies/prsSI'
elif vrbl == 'wind':
    var = 'uwnd'
    directory = '/Datasets/20thC_ReanV3/Dailies/prsSI'
elif vrbl == 'olr':
    var = 'ulwrf.sfc'
    directory = 'Datasets/20thC_ReanV3/Dailies/sfcFlxSI/'
elif vrbl == 't2m': 
    var = 'air'
    directory = '/Datasets/20thC_ReanV3/Dailies/2mSI'
ftp.cwd(directory)
filenames = ftp.nlst() # get filenames within the directory
filenames = [filename for filename in filenames if filename.endswith('.nc') and filename.startswith(var)]

out_path = Path(f'../Data/20CRv3/raw/{var}')
if vrbl == 'olr':
    out_path = Path(f'./raw/{vrbl}')
out_path.mkdir(parents=True, exist_ok=True)


existing_files = [p.name for p in sorted(out_path.glob('*.nc'))]
filenames = [f for f in filenames if f not in existing_files]

for filename in filenames:
    local_filename = out_path / filename
    with open(local_filename, 'wb+') as file:
        ftp.retrbinary('RETR '+ filename, file.write)
    file.close()
    if vrbl == 'pressure':
        xr.load_dataset(local_filename).sel(level=500).to_netcdf(local_filename, mode='w', engine ="netcdf4")
    elif vrbl == 't2m': 
        xr.load_dataset(local_filename).to_netcdf(local_filename, mode='w', engine ="netcdf4")
    elif vrbl == 'olr': 
        xr.load_dataset(local_filename).to_netcdf(local_filename, mode='w', engine ="netcdf4")
    else:
        xr.load_dataset(local_filename).sel(level=10).to_netcdf(local_filename, mode='w', engine ="netcdf4")


# Download files 1980 - 2015.
if vrbl == 'pressure':
    var = 'hgt'
    directory = '/Datasets/20thC_ReanV3/Dailies/prsMO'
elif vrbl == 't2m': 
    var = 'air'
    directory = '/Datasets/20thC_ReanV3/Dailies/2mMO'
elif vrbl == 'olr': 
    var = 'ulwrf.sfc'
    directory = 'Datasets/20thC_ReanV3/Dailies/sfcFlxMO/'

else:
    var = 'uwnd'
    directory = '/Datasets/20thC_ReanV3/Dailies/prsMO'
ftp.cwd(directory)
filenames = ftp.nlst() # get filenames within the directory
filenames = [filename for filename in filenames if filename.endswith('.nc') and filename.startswith(var)]

# check out path.
out_path = Path(f'../Data/20CRv3/raw/{var}')
if vrbl == 'olr':
    out_path = Path(f'./raw/{vrbl}')
out_path.mkdir(parents=True, exist_ok=True)

# check exsisting files.
existing_files = [p.name for p in sorted(out_path.glob('*.nc'))]
filenames = [f for f in filenames if f not in existing_files]

for filename in filenames:
    local_filename = out_path / filename
    with open(local_filename, 'wb+') as file:
        ftp.retrbinary('RETR '+ filename, file.write)
    file.close()
    
    if vrbl == 'pressure':
        xr.load_dataset(local_filename).sel(level=[500]).to_netcdf(local_filename, mode='w', engine ="netcdf4")
    elif vrbl == 't2m': 
        xr.load_dataset(local_filename).to_netcdf(local_filename, mode='w', engine ="netcdf4")
    elif vrbl == 'olr': 
        xr.load_dataset(local_filename).to_netcdf(local_filename, mode='w', engine ="netcdf4")
    else:
        xr.load_dataset(local_filename).sel(level=[10]).to_netcdf(local_filename, mode='w', engine ="netcdf4")

