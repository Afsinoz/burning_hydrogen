import pandas as pd
from pathlib import Path

directory_path = Path('../data')

chlor = pd.read_csv(directory_path / 'monthly_chlor_a.csv')
chlor.drop('Unnamed: 0', axis=1, inplace=True)
chlor['chlor_a'] = chlor['chlor_a'].astype('Float64')

sst = pd.read_csv(directory_path / 'mean_sst_subgrids.csv')
sst.rename({
    'lon': 'lon_rounded_up',
    'lat': 'lat_rounded_up'
}, axis=1, inplace=True)
sst['time'] = pd.to_datetime(sst['time'])
sst['year'] = sst['time'].dt.year
sst['month'] = sst['time'].dt.month
sst.drop('time', axis=1, inplace=True)
sst['lon_rounded_up'] = (sst['lon_rounded_up'] + 4.5).astype(int)
sst['lat_rounded_up'] = (sst['lat_rounded_up'] + 0.5).astype(int)
sst.replace('--', pd.NA, inplace=True)
sst['sst'] = sst['sst'].astype(dtype='Float64')

full_df = pd.merge(chlor, sst, on=[
                   'lon_rounded_up', 'lat_rounded_up', 'month', 'year'], how='inner')

depth = pd.read_csv(directory_path / 'Mean_Depth_Data.csv')
depth.drop('Unnamed: 0', axis=1, inplace=True)
depth.rename({
    'Longitude': 'lon_rounded_up',
    'Latitude': 'lat_rounded_up',
    'Mean_Elevation': 'elevation'
}, axis=1, inplace=True)
depth['lon_rounded_up'] = (depth['lon_rounded_up'] + 5).astype('int')
depth['lat_rounded_up'] = (depth['lat_rounded_up'] + 5).astype('int')
depth['elevation'] = depth['elevation'].astype('Float64')

full_df = pd.merge(full_df, depth, on=[
                   'lon_rounded_up', 'lat_rounded_up'], how='left')

sal_ox = pd.read_csv(directory_path / 'Mean_Temp_Salinity_Oxygen.csv')
sal_ox.drop(['Unnamed: 0', 't'], axis=1, inplace=True)
sal_ox.rename({
    'long': 'lon_rounded_up',
    'lat': 'lat_rounded_up',
    's': 'salinity',
}, axis=1, inplace=True)
sal_ox['lon_rounded_up'] = (sal_ox['lon_rounded_up'] + 5).astype(int)
sal_ox['lat_rounded_up'] = (sal_ox['lat_rounded_up'] + 5).astype(int)
sal_ox['salinity'] = sal_ox['salinity'].astype('Float64')
sal_ox['oxygen'] = sal_ox['oxygen'].astype('Float64')
full_df = pd.merge(full_df, sal_ox, on=[
    'year', 'month', 'lon_rounded_up', 'lat_rounded_up'], how='left')

full_df.to_csv(directory_path / 'full_df.csv')
