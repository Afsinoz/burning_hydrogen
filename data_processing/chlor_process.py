import pandas as pd
import numpy as np
from netCDF4 import Dataset
from pathlib import Path
import datetime

dates_to_process = list(pd.date_range('2024-01-01', '2024-01-03'))
date_strings = [st.strftime('%Y%m%d') for st in dates_to_process]

directory_path = Path.cwd() / 'output'
file_paths = [directory_path / 'AQUA_MODIS.{}.L3m.DAY.CHL.chlor_a.4km.nc'.format(date)
              for date in date_strings]
datasets = [Dataset(file_path) for file_path in file_paths]

lon_vals = np.arange(-100, 5, 5)
def lon_round_up(lon): return lon_vals[np.argmax(lon < lon_vals)]


lat_vals = np.arange(-15, 40, 5)
def lat_round_up(lat): return lat_vals[np.argmax(lat < lat_vals)]


dfs = [None] * len(datasets)

for i, ds in enumerate(datasets):
    date = pd.to_datetime(ds.time_coverage_start, yearfirst=True)
    fill_value = ds.variables['chlor_a']._FillValue

    chlor_a = np.array(ds.variables['chlor_a'])
    chlor_a = chlor_a.flatten()
    lon, lat = np.meshgrid(
        np.array(ds.variables['lon']), np.array(ds.variables['lat']))
    lon = lon.flatten()
    lat = lat.flatten()

    orig_df = pd.DataFrame({
        'lon': lon,
        'lat': lat,
        'chlor_a': chlor_a
    })

    local_df = orig_df.loc[(-100 <= orig_df.lon) & (orig_df.lon <= 0)
                           & (-15 <= orig_df.lat) & (orig_df.lat <= 35)]

    local_df = local_df.replace(fill_value, value=pd.NA)

    local_df['lon_rounded_up'] = local_df['lon'].apply(lon_round_up)
    local_df['lat_rounded_up'] = local_df['lat'].apply(lat_round_up)

    df = pd.DataFrame(local_df.groupby(
        ['lon_rounded_up', 'lat_rounded_up'])['chlor_a'].mean())
    df['date'] = date
    df = df[['date', 'chlor_a']]

    df.reset_index(inplace=True)

    dfs[i] = df

full_df = pd.concat(dfs, axis=0)
full_df.reset_index(inplace=True)
full_df = full_df.drop(labels='index', axis=1)
full_df['year'] = full_df['date'].dt.year
full_df['month'] = full_df['date'].dt.month

monthly_df = pd.DataFrame(full_df.groupby(
    ['year', 'month', 'lon_rounded_up', 'lat_rounded_up'])['chlor_a'].mean())
monthly_df.reset_index(inplace=True)

full_df.to_csv('full_chlor_a.csv')
monthly_df.to_csv('monthly_chlor_a.csv')
