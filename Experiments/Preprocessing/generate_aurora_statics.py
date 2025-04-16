import cdsapi

exd = os.path.dirname(os.path.abspath(__file__))
cfd = Path(exd).parent.absolute()
config = yaml.load(open(f"{cdf}/config/{vrbl}_data.yml"), Loader=yaml.FullLoader)

root = Path(cdf).parent.absolute()

download_path = Path(f"{root}/Data/ERA5/datasets/")

c = cdsapi.Client()

download_path = download_path.expanduser()
download_path.mkdir(parents=True, exist_ok=True)

# Download the static variables.
if not (download_path / "static.nc").exists():
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "geopotential",
                "land_sea_mask",
                "soil_type",
            ],
            "year": "2023",
            "month": "01",
            "day": "01",
            "time": "00:00",
            "format": "netcdf",
        },
        str(download_path / "static.nc"),
    )
print("Static variables downloaded!")

statics= xr.open_dataset(download_path / "static.nc")

# import xesmf as xe
# regridding
resolution = '1.40625'
grid_path = Path(f'{root}/Data_S2S/weatherbench_grid.nc')
if grid_path.exists():
    grid = xr.load_dataset(grid_path)
    regridder = xe.Regridder(statics.slt, grid, "bilinear")
    ds = regridder(statics.lsm)
else:
    print('No grid found. Skipping regridding.')

ds.to_netcdf(download_path / "static_regrid.nc")

static_vars_ds = ds
static_vars_ds = static_vars_ds.assign_coords(lon= static_vars_ds.lon%360)
static_vars_ds = static_vars_ds.sortby('lon')
surf_vars_ds = xr.open_dataarray(download_path / f"olr__1.40625deg_1980-2023_northern_hemi_2d.nc", engine="netcdf4")
surf_vars_ds = surf_vars_ds.assign_coords(lon= surf_vars_ds.lon%360)
surf_vars_ds = surf_vars_ds.sortby('lon')

lat_min = surf_vars_ds.lat.min().values
lat_max = surf_vars_ds.lat.max().values
static_vars_ds = static_vars_ds.sel(**{"lat":slice(lat_min,lat_max)})
static_vars_ds.to_netcdf(download_path / "static_regrid_northern_hemi.nc")
