import cdsapi
from argparse import ArgumentParser
import os


# set variable downloaded from the ERA5
parser = ArgumentParser()

parser.add_argument("--vrbl", type=str, default='pressure')
args = parser.parse_args()

vrbl = args.vrbl

c = cdsapi.Client()
target_folder = '../Data/ERA5/raw/'

if 'olr' in vrbl:
    years=['1980', '1981', '1982',
                '1983', '1984', '1985',
                '1986', '1987', '1988',
                '1989', '1990', '1991',
                '1992', '1993', '1994',
                '1995', '1996', '1997',
                '1998', '1999', '2000',
                '2001', '2002', '2003',
                '2004', '2005', '2006',
                '2007', '2008', '2009',
                '2010', '2011', '2012',
                '2013', '2014', '2015',
                '2016', '2017', '2018',
                '2019', '2020', '2021',
                '2022', '2023',]
    for yrs in years:
        if not os.path.exists(f'./raw/olr_northern/olr_{yrs}.nc'): 
            c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': 'top_net_thermal_radiation',
                    'year': yrs,
                'month': [
                '01', '02', '03','04',
                '10','11', '12',],
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '06:00',
                        '12:00',
                        '18:00',
                    ],
                    'area': [
                        90, -180, -20,
                        180,
                    ],
                },
                f'{target_folder}/olr_north_7m/olr_{yrs}.nc')


elif 'pressure' in vrbl:
    if not os.path.exists(f'./raw/geopotential_z500.nc'):
        dataset = 'reanalysis-era5-pressure-levels'
        request = {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': 'geopotential',
                'pressure_level': '500',
                'year': [
                    '1980', '1981', '1982',
                    '1983', '1984', '1985',
                    '1986', '1987', '1988',
                    '1989', '1990', '1991',
                    '1992', '1993', '1994',
                    '1995', '1996', '1997',
                    '1998', '1999', '2000',
                    '2001', '2002', '2003',
                    '2004', '2005', '2006',
                    '2007', '2008', '2009',
                    '2010', '2011', '2012',
                    '2013', '2014', '2015',
                    '2016', '2017', '2018',
                    '2019', '2020', '2021',
                    '2022', '2023',
                ],
                'month': [
                    '01', '02', '03','04',
                    '10','11', '12',
                ],
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'area': [
                            90, -180, 0,
                            180,
                        ],
                'time': [
                    '00:00', '06:00', '12:00',
                    '18:00'
                ],
            }
        c.retrieve(
            dataset,
            request,
            f'{target_folder}geopotential_z500.nc')

elif 'wind' in vrbl:
    if not os.path.exists(f'./raw/uwind_10.nc'):
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': 'u_component_of_wind',
                'pressure_level': '10',
                'year': [
                    '1980', '1981', '1982',
                    '1983', '1984', '1985',
                    '1986', '1987', '1988',
                    '1989', '1990', '1991',
                    '1992', '1993', '1994',
                    '1995', '1996', '1997',
                    '1998', '1999', '2000',
                    '2001', '2002', '2003',
                    '2004', '2005', '2006',
                    '2007', '2008', '2009',
                    '2010', '2011', '2012',
                    '2013', '2014', '2015',
                    '2016', '2017', '2018',
                    '2019', '2020', '2021',
                    '2022', '2023',
                ],
                'month': [
                '01', '02', '03','04',
                    '10','11', '12',],
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '06:00', '12:00',
                    '18:00',
                ],
            },
            f'{target_folder}/uwind_10.nc')
