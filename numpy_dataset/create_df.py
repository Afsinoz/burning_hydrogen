import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle

# Number of lag features and lead features to create in the dataset
num_lag = 10
num_lead = 30

# Pull daily O2 data
o2_dir = Path('../data/daily_data/o2_raw')
o2_files = sorted(os.listdir(o2_dir))
o2_files = [file for file in o2_files if file[0] != '.']
o2_dfs = [pd.read_csv(o2_dir / file) for file in o2_files]
o2_df = pd.concat(o2_dfs, axis=0)
o2_df.drop('Unnamed: 0', inplace=True, axis=1)

# Group O2 data by 'date', 'lon_rounded_up', 'lat_rounded_up',
# take mean over 'depth'
o2_df = o2_df.groupby(
    by=['date', 'lon_rounded_up', 'lat_rounded_up']).mean()
o2_df.reset_index(inplace=True)
o2_df.drop('depth', axis=1, inplace=True)

# Pull daily geochemical data
geo_dir = Path('../data/daily_data/geo_raw')
geo_files = sorted(os.listdir(geo_dir))
geo_files = [file for file in geo_files if file[0] != '.']
geo_files.remove('readme.txt')
geo_dfs = [pd.read_csv(geo_dir / file) for file in geo_files]
geo_df = pd.concat(geo_dfs, axis=0)

# Convert 'longitude_bins' to 'lon_rounded_up', 'latitude_bins'
# to 'lat_rounded_up', 'time' to 'date'
geo_df['lon_rounded_up'] = 5 * geo_df['longitude_bins'] - 95
geo_df['lat_rounded_up'] = 5 * geo_df['latitude_bins'] - 10
geo_df.drop(['longitude_bins', 'latitude_bins'], axis=1, inplace=True)
geo_df = geo_df[geo_df.time <= '2024-03-31']
geo_df.rename({'time': 'date'}, axis=1, inplace=True)

# Group geochemical data by 'date', 'lon_rounded_up', 'lat_rounded_up',
# take mean over 'depth'
geo_df = geo_df.groupby(
    by=['date', 'lon_rounded_up', 'lat_rounded_up']).mean()
geo_df.reset_index(inplace=True)
geo_df.drop('depth', axis=1, inplace=True)

# Pull daily salinity data, concatenate into single dataframe
sal_dir = Path('../data/daily_data/Salinity_data')
sal_files = sorted(os.listdir(sal_dir))
sal_files = [file for file in sal_files if file[0] != '.']
sal_dfs = [pd.read_csv(sal_dir / file) for file in sal_files]
for df in sal_dfs:
    df.rename({
        'Salinity': 'salinity'
    }, inplace=True, axis=1)
sal_df = pd.concat(sal_dfs, axis=0)

# Convert 'longitude' to 'lon_rounded_up', 'latitude' to 'lat_rounded_up',
# 'time' to 'date'
sal_df = sal_df[(sal_df.longitude != 0.0) & (sal_df.latitude != 35.0)]
sal_df.longitude = (sal_df.longitude + 5).astype(int)
sal_df.latitude = (sal_df.latitude + 5).astype(int)
sal_df.rename({
    'time': 'date',
    'longitude': 'lon_rounded_up',
    'latitude': 'lat_rounded_up'
}, inplace=True, axis=1)

# Group salinity data by 'date', 'lon_rounded_up', 'lat_rounded_up',
# take mean over 'depth'
sal_df = sal_df.groupby(
    by=['date', 'lon_rounded_up', 'lat_rounded_up']).mean()
sal_df.reset_index(inplace=True)
sal_df.drop('depth', axis=1, inplace=True)

# Pull daily temperature data
temp_dir = Path('../data/daily_data/Temp')
temp_files = sorted(os.listdir(temp_dir))
temp_files = [file for file in temp_files if file[0] != '.']
temp_dfs = [pd.read_csv(temp_dir / file) for file in temp_files]
for df in temp_dfs:
    df.rename({
        'Temp': 'temp'
    }, inplace=True, axis=1)
temp_df = pd.concat(temp_dfs, axis=0)

# Convert 'longitude' to 'lon_rounded_up', 'latitude' to 'lat_rounded_up',
# 'time' to 'date'
temp_df = temp_df[(temp_df.longitude != 0.0) & (temp_df.latitude != 35.0)]
temp_df.longitude = (temp_df.longitude + 5).astype(int)
temp_df.latitude = (temp_df.latitude + 5).astype(int)
temp_df.rename({
    'time': 'date',
    'longitude': 'lon_rounded_up',
    'latitude': 'lat_rounded_up'
}, inplace=True, axis=1)

# Group temperature data by 'date', 'lon_rounded_up', 'lat_rounded_up',
# take mean over 'depth'
temp_df = temp_df.groupby(
    by=['date', 'lon_rounded_up', 'lat_rounded_up']).mean()
temp_df.reset_index(inplace=True)
temp_df.drop('depth', axis=1, inplace=True)

# Pull depth data
depth_df = pd.read_csv('../data/Mean_Depth_Data.csv')
depth_df.drop('Unnamed: 0', inplace=True, axis=1)
depth_df['Longitude'] = (depth_df['Longitude'] + 5).astype(int)
depth_df['Latitude'] = (depth_df['Latitude'] + 5).astype(int)
depth_df.rename({
    'Longitude': 'lon_rounded_up',
    'Latitude': 'lat_rounded_up',
    'Mean_Elevation': 'elevation'
}, axis=1, inplace=True)

# Restrict each dataframe to the same date range
o2_df = o2_df[(o2_df.date >= '2021-11-30') & (o2_df.date <= '2024-03-31')]
geo_df = geo_df[(geo_df.date >= '2021-11-30') & (geo_df.date <= '2024-03-31')]
sal_df = sal_df[(sal_df.date >= '2021-11-30') & (sal_df.date <= '2024-03-31')]
temp_df = temp_df[(temp_df.date >= '2021-11-30') &
                  (temp_df.date <= '2024-03-31')]


# Merge all dataframes
full_df = pd.merge(o2_df, geo_df, on=['date', 'lon_rounded_up', 'lat_rounded_up'],
                   how='outer')
full_df = pd.merge(full_df, sal_df, on=['date', 'lon_rounded_up', 'lat_rounded_up'],
                   how='outer')
full_df = pd.merge(full_df, temp_df, on=['date', 'lon_rounded_up', 'lat_rounded_up'],
                   how='outer')
full_df = pd.merge(full_df, depth_df, on=['lon_rounded_up', 'lat_rounded_up'],
                   how='left')

# Features to use in creating lag and lead features
the_features = ['chl', 'no3', 'po4', 'si', 'salinity', 'temp', 'o2']

# Create copy of full_df with multi-index on 'date', 'lon_rounded_up',
# 'lat_rounded_up'
full_df_multi = full_df.set_index(['date', 'lon_rounded_up', 'lat_rounded_up'])

# Add lag features to full_df_multi and keep track of the created
# lag features in lag_features
lag_features = []
for i in range(1, num_lag + 1):
    for feature in the_features:
        lag_features.append(feature + f'_lag_{i}')
        full_df_multi[feature + f'_lag_{i}'] = \
            full_df_multi.groupby(level=[1, 2])[feature].shift(i)

# Add lead features to full_df_multi and keep track of the created
# lead features in lead_features
lead_features = []
for i in range(1, num_lead + 1):
    for feature in the_features:
        lead_features.append(feature + f'_lead_{i}')
        full_df_multi[feature + f'_lead_{i}'] = \
            full_df_multi.groupby(level=[1, 2])[feature].shift(-i)

# Remove any dates that are missing any corresponding lag or lead features
dates_to_remove = pd.date_range('2021-11-30', periods=num_lag).tolist() + \
    pd.date_range(end='2024-03-31', periods=num_lead).tolist()
dates_to_remove = [ts.strftime('%Y-%m-%d') for ts in dates_to_remove]
full_df_multi.drop(level=0, axis=0, labels=dates_to_remove, inplace=True)

# Reset index of full_df_multi and save it to a csv file
full_df_multi.reset_index(inplace=True)
full_df_multi.to_csv('dataset.csv')

# base_date = pd.to_datetime(full_df_multi.date[0])
# full_df_multi.date = pd.to_datetime(full_df_multi.date) - base_date
# full_df_multi.date = full_df_multi.date.dt.days.astype(int)

# dates = full_df_multi.date.unique()

# dates_train, dates_test_full = train_test_split(
#     dates, test_size=0.4, shuffle=False)
# dates_valid, dates_test = train_test_split(
#     dates_test_full, test_size=0.5, shuffle=False)

# X = full_df_multi.drop(lead_features, axis=1)
# y = full_df_multi[['date'] + lead_features]

# X_train = X.loc[(X.date >= dates_train[0]) & (X.date <= dates_train[-1])]
# y_train = y.loc[(X.date >= dates_train[0]) & (X.date <= dates_train[-1])]
# X_valid = X.loc[(X.date >= dates_valid[0]) & (X.date <= dates_valid[-1])]
# y_valid = y.loc[(X.date >= dates_valid[0]) & (X.date <= dates_valid[-1])]
# X_test = X.loc[(X.date >= dates_test[0]) & (X.date <= dates_test[-1])]
# y_test = y.loc[(X.date >= dates_test[0]) & (X.date <= dates_test[-1])]

# X_train.to_csv('./splits/X_train.csv')
# y_train.to_csv('./splits/y_train.csv')
# X_valid.to_csv('./splits/X_valid.csv')
# y_valid.to_csv('./splits/y_valid.csv')
# X_test.to_csv('./splits/X_test.csv')
# y_test.to_csv('./splits/y_test.csv')
