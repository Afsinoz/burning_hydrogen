import numpy as np
import netCDF4
import sys

#   Download the GEBCO_2024 Grid netCDF file  
#   https://www.gebco.net/data_and_products/gridded_bathymetry_data/
#
#
#   Run script as GEBCO_ElevationFunction.py arg1 arg2
#
#
#   Elevation (relative to mean sea level)
#
#
#   Reference: https://github.com/MikeHudgell/GEBCO
#
#
def geo_idx(degd, degd_array):
    #   
    #
    #   Returns index of the closest degree in the array
    #   degd = degrees decimal
    #   degd_array = arracy of degrees decimal values
    #
    #
    geo_index = (np.abs(degd_array - degd)).argmin()
    return geo_index

def open_GEBCO_file(filepath):
    #
    #
    #   Returns dataset
    #   filepath = filepath to GEBCO file
    #
    #
    NetCDF_dataset = netCDF4.Dataset(filepath)
    lats = NetCDF_dataset.variables['lat'][:]
    lons = NetCDF_dataset.variables['lon'][:]
    return NetCDF_dataset, lats, lons

def get_GEBCO_info(dataset):
    #
    #
    #   Prints metadata in GEBCO file
    #
    #
    print(dataset.data_model)

    for attr in dataset.ncattrs():
        print(attr, '=', getattr(dataset, attr))

    print(dataset.variables)
    return

def get_elevation(lat, lon):
    #
    #
    #   Returns the elevation from the latitude and longitude
    #
    #
    lat_index = geo_idx(lat, lats)
    lon_index = geo_idx(lon, lons)
    return gebco.variables['elevation'][lat_index, lon_index]
#
#
#   Opens file
gebco, lats, lons  = open_GEBCO_file('GEBCO_2024.nc')
#
#   Allows script to take arguments
if __name__ == "__main__":
    lat = float(sys.argv[1])
    lon = float(sys.argv[2])
    print(get_elevation(lat,lon))
#
#
