import pandas as pd
import numpy as np

df = pd.read_csv('./splits/dataset_reordered.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

df_multi = df.set_index(['date', 'lon_rounded_up', 'lat_rounded_up'])

features = ['elevation', 'o2', 'chl', 'no3', 'po4', 'si', 'salinity', 'temp']

num_x = df['lon_rounded_up'].nunique()
num_y = df['lat_rounded_up'].nunique()
num_dates = df['date'].nunique()
num_channels = len(features)

num_lag = 10
num_lead = 30
num_days = num_lag + num_lead + 1

output = np.empty(shape=(num_dates, num_days, num_channels, num_y, num_x))
for i, date in enumerate(pd.date_range('2021-12-10', '2024-03-01').strftime('%Y-%m-%d')):
    for j, lat in enumerate(range(-10, 40, 5)):
        for k, lon in enumerate(range(-95, 5, 5)):
            output[i, :, :, j, k] = df_multi.loc[date, lon,
                                                 lat].to_numpy().reshape(
                num_days, num_channels)

X = output[:, :11, :, :, :]
y = output[:, 11:, :, :, :]

np.save('./splits/dataset.npy', output)
np.save('./splits/X.npy', X)
np.save('./splits/y.npy', y)
