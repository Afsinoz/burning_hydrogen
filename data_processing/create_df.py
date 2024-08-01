import pandas as pd
import numpy as np
from netCDF4 import Dataset


def nc_to_dfs(files):
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
    lat_1_dim = np.array(ds.variables['lat'])
    lon_1_dim = np.array(ds.variables['lon'])
    lat, lon = np.meshgrid(lat_1_dim, lon_1_dim)

    return chlor_a, lat, lon


files = ['AQUA_MODIS.20240101.L3m.DAY.CHL.chlor_a.4km.nc',
         'AQUA_MODIS.20240102.L3m.DAY.CHL.chlor_a.4km.nc',
         'AQUA_MODIS.20240103.L3m.DAY.CHL.chlor_a.4km.nc',
         'AQUA_MODIS.20020704.L3m.DAY.CHL.chlor_a.4km.nc']

print(nc_to_dfs(files))
