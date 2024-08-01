# Write file names for MODIS chlorophyll-a data for a specified
# date range to a text file.
# Usage: bash write_dates.sh start_date end_date file_to_write_to
# start_date and end_date have format yyyymmdd.

d="$1"

until [[ "$d" > "$2" ]]
do
	echo "https://oceandata.sci.gsfc.nasa.gov/getfile/AQUA_MODIS.$d.L3m.DAY.CHL.chlor_a.4km.nc" >> $3
	d=`gdate --date="$d +1 day" +%Y%m%d`
done

