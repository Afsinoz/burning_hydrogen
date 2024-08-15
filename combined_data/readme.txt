Here we have dataset into smaller chunks based on time intervals, such as every 6 months.
Starting from 2021-11-01 and every 6 months interval we have data until 2024-05-31. 

longitude_bins are index 0 to 19. 
e.g. longitude_bins 0 is mean of values from -100 to -96
longitude_bins 1 is mean of values from -95 to -91 and so on.

latitude_bins are index from 0 to 9.
just like longitude.

# Define the latitude and longitude bins
lat_bins = np.arange(-15, 36, 5)  # From -15 to 35 with step 5
lon_bins = np.arange(-100, 1, 5)  # From -100 to 0 with step 5