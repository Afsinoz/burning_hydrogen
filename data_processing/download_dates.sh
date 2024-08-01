# Download MODIS chlorophyll-a files for a specified date range
# to a directory.
# Usage: bash download_dates.sh start_date end_date filename directory
# start_date and end_date have format yyyymmdd.
# filename is the name of a file to write the MODIS file names to.
# directory is the name of a directory to save the MODIS .nc files to.

bash write_dates.sh "$1" "$2" "$3"

python3 obdaac_download.py -v --filelist "$3" --odir "$4" 
