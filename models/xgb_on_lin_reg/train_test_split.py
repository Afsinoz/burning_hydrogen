import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

num_lag = 2
num_lead = 2

o2_dir = Path('../data/daily_data/o2_raw')
o2_files = sorted(os.listdir(o2_dir))
o2_files = [file for file in o2_files if file[0] != '.']
o2_dfs = [pd.read_csv(o2_dir / file) for file in o2_files]
o2_df = pd.concat(o2_dfs, axis=0)
o2_df.drop('Unnamed: 0', inplace=True, axis=1)

geo_dir = Path('../data/daily_data/geo_raw')
geo_files = sorted(os.listdir(geo_dir))
geo_files = [file for file in geo_files if file[0] != '.']
geo_files.remove('readme.txt')
geo_dfs = [pd.read_csv(geo_dir / file) for file in geo_files]
geo_df = pd.concat(geo_dfs, axis=0)

geo_df['lon_rounded_up'] = 5 * geo_df['longitude_bins'] - 95
geo_df['lat_rounded_up'] = 5 * geo_df['latitude_bins'] - 10
geo_df.drop(['longitude_bins', 'latitude_bins'], axis=1, inplace=True)
geo_df = geo_df[geo_df.time <= '2024-03-31']
geo_df.rename({'time': 'date'}, axis=1, inplace=True)

o2_df = o2_df[o2_df.date >= '2021-11-30']

full_df = pd.merge(o2_df, geo_df, on=['date', 'lon_rounded_up', 'lat_rounded_up', 'depth'],
                   how='outer')

depth_df = pd.read_csv('../data/Mean_Depth_Data.csv')
depth_df.drop('Unnamed: 0', inplace=True, axis=1)
depth_df['Longitude'] = (depth_df['Longitude'] + 5).astype(int)
depth_df['Latitude'] = (depth_df['Latitude'] + 5).astype(int)
depth_df.rename({
    'Longitude': 'lon_rounded_up',
    'Latitude': 'lat_rounded_up',
    'Mean_Elevation': 'elevation'
}, axis=1, inplace=True)

full_df = pd.merge(full_df, depth_df, on=['lon_rounded_up', 'lat_rounded_up'],
                   how='left')

full_df = full_df.groupby(
    by=['date', 'lon_rounded_up', 'lat_rounded_up']).mean()
full_df.reset_index(inplace=True)
full_df.drop('depth', axis=1, inplace=True)

the_features = ['o2', 'chl', 'no3', 'po4', 'si']

full_df_multi = full_df.set_index(['date', 'lon_rounded_up', 'lat_rounded_up'])
lag_features = []

for i in range(1, num_lag + 1):
    for feature in the_features:
        lag_features.append(feature + f'_lag_{i}')
        full_df_multi[feature + f'_lag_{i}'] = \
            full_df_multi.groupby(level=[1, 2])[feature].shift(i)

lead_features = []

for i in range(1, num_lead + 1):
    for feature in the_features:
        lead_features.append(feature + f'_lead_{i}')
        full_df_multi[feature + f'_lead_{i}'] = \
            full_df_multi.groupby(level=[1, 2])[feature].shift(-i)

dates_to_remove = pd.date_range('2021-11-30', periods=num_lag).tolist() + \
    pd.date_range(end='2024-03-31', periods=num_lead).tolist()
dates_to_remove = [ts.strftime('%Y-%m-%d') for ts in dates_to_remove]

full_df_multi.drop(level=0, axis=0, labels=dates_to_remove, inplace=True)
full_df_multi.reset_index(inplace=True)
full_df_multi.dropna(axis=0, how='any', inplace=True)

base_date = pd.to_datetime(full_df_multi.date[0])
full_df_multi.date = pd.to_datetime(full_df_multi.date) - base_date
full_df_multi.date = full_df_multi.date.dt.days.astype(int)

dates = full_df_multi.date.unique()

dates_train, dates_test_full = train_test_split(
    dates, test_size=0.4, shuffle=False)
dates_valid, dates_test = train_test_split(
    dates_test_full, test_size=0.5, shuffle=False)

X = full_df_multi.drop(lead_features, axis=1)
y = full_df_multi[['date'] + lead_features]

X_train = X.loc[(X.date >= dates_train[0]) & (X.date <= dates_train[-1])]
y_train = y.loc[(X.date >= dates_train[0]) & (X.date <= dates_train[-1])]
X_valid = X.loc[(X.date >= dates_valid[0]) & (X.date <= dates_valid[-1])]
y_valid = y.loc[(X.date >= dates_valid[0]) & (X.date <= dates_valid[-1])]
X_test = X.loc[(X.date >= dates_test[0]) & (X.date <= dates_test[-1])]
y_test = y.loc[(X.date >= dates_test[0]) & (X.date <= dates_test[-1])]

for df in [y_train, y_valid, y_test]:
    df.drop('date', axis=1, inplace=True)

X_train.to_csv('./splits/X_train.csv')
y_train.to_csv('./splits/y_train.csv')
X_valid.to_csv('./splits/X_valid.csv')
y_valid.to_csv('./splits/y_valid.csv')
X_test.to_csv('./splits/X_test.csv')
y_test.to_csv('./splits/y_test.csv')
