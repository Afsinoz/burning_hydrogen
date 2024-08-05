import pandas as pd
import numpy as np
from netCDF4 import Dataset
from pathlib import Path


def nc_to_arr(files):
    """
    Read in a list of .nc files for chlorophyll-a concentration on distinct days
    from AQUA-MODIS dataset. Return a 3d numpy array chlor_a recording the
    chlorophyll-a concentration by latitude and longitude and day. Return
    two 2d numpy arrays lat, lon recording the latitude and longitude for
    the entries of chlor_a.
    """
    directory_path = Path.cwd()
    file_paths = [directory_path / file_name for file_name in files]
    datasets = [Dataset(file_path) for file_path in file_paths]
    chlor_a = np.array([ds.variables['chlor_a'][:, :].data for ds in datasets])

    ds = datasets[0]
    lon_1_dim = np.array(ds.variables['lon'])
    lat_1_dim = np.array(ds.variables['lat'])
    lon, lat = np.meshgrid(lon_1_dim, lat_1_dim)

    return chlor_a, lon, lat


def grid(arr, lon_subdiv, lat_subdiv):
    """
    Take each timestamp in arr, split it into chunks where there are lon_subdiv
    chunks in the longitude direction and lat_subdiv chunks in the latitude
    direction. For each timestamp, take the mean over each chunk.
    """
    splits = np.array([np.split(sub, indices_or_sections=lon_subdiv, axis=2)
                       for sub in np.split(arr, indices_or_sections=lat_subdiv, axis=1)])
    gridded = np.mean(splits, axis=(3, 4))
    gridded = np.transpose(gridded, (2, 0, 1))
    return gridded
