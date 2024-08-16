import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

o2_dir = Path('../data/daily_data/o2_raw')
o2_files = sorted(os.listdir(o2_dir))

o2_files.remove('.DS_Store')
o2_dfs = [pd.read_csv(o2_dir / file) for file in o2_files]
o2_df = pd.concat(o2_dfs, axis=0)
o2_df.drop('Unnamed: 0', inplace=True, axis=1)

geo_dir = Path('../data/daily_data/geo_raw')
geo_files = sorted(os.listdir(geo_dir))
geo_files.remove('.DS_Store')
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

full_df = full_df.groupby(
    by=['date', 'lon_rounded_up', 'lat_rounded_up']).mean()
full_df.reset_index(inplace=True)
full_df.drop('depth', axis=1, inplace=True)

the_features = ['o2', 'chl', 'no3', 'po4', 'si']

full_df_multi = full_df.set_index(['date', 'lon_rounded_up', 'lat_rounded_up'])
lag_features = []

for feature in the_features:
    for i in range(1, 31):
        lag_features.append(feature + f'_lag_{i}')
        full_df_multi[feature + f'_lag_{i}'] = \
            full_df_multi.groupby(level=[1, 2])[feature].shift(i)

lead_features = []
for feature in the_features:
    for i in range(1, 31):
        lead_features.append(feature + f'_lead_{i}')
        full_df_multi[feature + f'_lead_{i}'] = \
            full_df_multi.groupby(level=[1, 2])[feature].shift(-i)

dates_to_remove = pd.date_range('2021-11-30', periods=30).tolist() + \
    pd.date_range(end='2024-03-31', periods=30).tolist()
dates_to_remove = [ts.strftime('%Y-%m-%d') for ts in dates_to_remove]

full_df_multi.drop(level=0, axis=0, labels=dates_to_remove, inplace=True)

dates = full_df_multi.index.levels[0].to_list()
dates = [date for date in dates if date not in dates_to_remove]

dates_train, dates_test_full = train_test_split(
    dates, test_size=0.4, shuffle=False)
dates_valid, dates_test = train_test_split(
    dates_test_full, test_size=0.5, shuffle=False)

full_df_multi.dropna(axis=0, how='all', inplace=True)
X = full_df_multi.drop(lead_features, axis=1)
y = full_df_multi[lead_features]

X_train = X.loc[dates_train]
y_train = y.loc[dates_train]
X_valid = X.loc[dates_valid]
y_valid = y.loc[dates_valid]
X_test = X.loc[dates_test]
y_test = y.loc[dates_test]

for df in [X_train, y_train, X_valid, y_valid, X_test, y_test]:
    df.reset_index(inplace=True)

base_date = pd.to_datetime(dates[0])

for df in [X_train, X_valid, X_test]:
    df.date = pd.to_datetime(df.date) - base_date
    df.date = df.date.dt.days

for df in [y_train, y_valid, y_test]:
    df.drop(['date', 'lon_rounded_up', 'lat_rounded_up'], axis=1, inplace=True)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

for i in range(30):
    err = np.sqrt(mean_squared_error(y_pred[:, i], y_valid.iloc[:, i]))
    print(f'Day: {i}')
    print(f'Average oxygen prediction error: {err}')
