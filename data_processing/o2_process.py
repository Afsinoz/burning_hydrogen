import pandas as pd
import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime, timedelta
import os

directory_path = Path(
    '/Users/alexanderrasmussen/library/cloudstorage/GoogleDrive-ajrmath@gmail.com/My Drive/erdos/copernicus_o2')

file_name = 'cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m_o2_100.00W-0.00E_15.00S-35.00N_0.49-47.37m_2021-11-01-2024-03-31.nc'

ds = Dataset(directory_path / file_name)

init_date = datetime(1950, 1, 1, 0, 0, 0)

o2 = np.array(ds.variables['o2'])

lon = np.array(ds.variables['longitude'])
lat = np.array(ds.variables['latitude'])
time = np.array(ds.variables['time'])
depth = np.array(ds.variables['depth'])

depth, lat, lon = np.meshgrid(depth, lat, lon, indexing='ij')
depth = depth.flatten()
lon = lon.flatten()
lat = lat.flatten()

dfs = [None] * time.shape[0]

fill_value = ds.variables['o2']._FillValue

lon_vals = np.arange(-95, 5, 5)
def lon_round_up(lon): return lon_vals[np.argmax(lon <= lon_vals)]


lat_vals = np.arange(-10, 40, 5)
def lat_round_up(lat): return lat_vals[np.argmax(lat <= lat_vals)]


for i, (ox, hours) in enumerate(zip(o2, time)):
    cur_date = init_date + timedelta(hours=int(hours))
    print(cur_date)

    ox = ox.flatten()
    df = pd.DataFrame({
        'lon': lon,
        'lat': lat,
        'depth': depth,
        'o2': ox
    })

    df.replace(fill_value, value=pd.NA, inplace=True)

    df['lon_rounded_up'] = df['lon'].apply(lon_round_up)
    df['lat_rounded_up'] = df['lat'].apply(lat_round_up)

    df = pd.DataFrame(df.groupby(
        ['lon_rounded_up', 'lat_rounded_up', 'depth'])['o2'].mean())
    df['date'] = cur_date

    df.reset_index(inplace=True)

    df = df[['date', 'lon_rounded_up', 'lat_rounded_up', 'depth', 'o2']]

    dfs[i] = df
    df.to_csv('../data/o2_raw/o2-{}.csv'.format(cur_date))

full_df = pd.concat(dfs, axis=0)
full_df.reset_index(inplace=True)
full_df = full_df.drop(labels='index', axis=1)
full_df['year'] = full_df['date'].dt.year
full_df['month'] = full_df['date'].dt.month

full_df.to_csv('../data/full_o2.csv')
