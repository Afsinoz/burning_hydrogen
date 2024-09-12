import pandas as pd
import numpy as np

# Reorder dataset.csv in order to create a numpy array from it

# Load the dataset
df = pd.read_csv('./splits/dataset.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# Rename lag and lead features
chem_features = ['o2', 'chl', 'no3', 'po4', 'si', 'salinity', 'temp']
df.rename({feature: feature + f'_t_0' for feature in chem_features},
          axis=1, inplace=True)
df.rename({feature + f'_lag_{i}': feature + f'_t_{-i}' for i in range(1, 11)
           for feature in chem_features}, axis=1, inplace=True)
df.rename({feature + f'_lead_{i}': feature + f'_t_{i}' for i in range(1, 31)
           for feature in chem_features}, axis=1, inplace=True)

# Add lag and lead features for elevation
for i in range(-10, 31):
    df[f'elevation_t_{i}'] = df['elevation']
df.drop('elevation', axis=1, inplace=True)

# Collect all features to appear in the numpy array
features = ['elevation'] + chem_features

# Reorder the columns of df and save to a csv file
df = df[['date', 'lon_rounded_up', 'lat_rounded_up'] +
        [feature + f'_t_{i}' for i in range(-10, 31) for feature in features]]
df.to_csv('./splits/dataset_reordered.csv')
