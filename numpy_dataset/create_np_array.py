import pandas as pd
import numpy as np

# Create a numpy array from the dataframe dataset_reordered.csv

# Open dataset_reordered.csv
df = pd.read_csv('dataset_reordered.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# Create copy of df with multi-index
df_multi = df.set_index(['date', 'lon_rounded_up', 'lat_rounded_up'])

# Variables to be used in numpy array
features = ['elevation', 'o2', 'chl', 'no3', 'po4', 'si', 'salinity', 'temp']

# num_x = number of x-coordinates per date
num_x = df['lon_rounded_up'].nunique()
# num_y = number of y-coordinates per date
num_y = df['lat_rounded_up'].nunique()
# num_dates = total number of dates in dataset
num_dates = df['date'].nunique()
# num_channels = number of variables measured per date
num_channels = len(features)
num_lag = 10
num_lead = 30
# num_days = total number of days of lag, lead, plus current day, per date
num_days = num_lag + num_lead + 1

# Create output numpy array of shape
# num_dates x num_days x num_channels x num_y x num_x.
# For each triple of (date, feature, day), where feature is in features and
# day indicates the number of days to lag or lead, (with day=0 indicating lag=10
# and day=41 indicating lead=10) we have a 2-dimensional numpy array indicating
# the values of that feature on that date over our region lagged or leaded by the
# specified number of days.
output = np.empty(shape=(num_dates, num_days, num_channels, num_y, num_x))
for i, date in enumerate(pd.date_range('2021-12-10', '2024-03-01').strftime('%Y-%m-%d')):
    for j, lat in enumerate(range(-10, 40, 5)):
        for k, lon in enumerate(range(-95, 5, 5)):
            output[i, :, :, j, k] = df_multi.loc[date, lon,
                                                 lat].to_numpy().reshape(
                num_days, num_channels)

# The input X will consist of the lag features on each date, plus the data for
# the current day. The output y will consist of the lead features on each date.
X = output[:, :11, :, :, :]
y = output[:, 11:, :, :, :]

# Save the arrays.
np.save('dataset.npy', output)
np.save('X.npy', X)
np.save('y.npy', y)
