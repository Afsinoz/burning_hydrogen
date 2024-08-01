d="$1"

until [[ "$d" > "$2" ]]
do
	echo "https://oceandata.sci.gsfc.nasa.gov/getfile/AQUA_MODIS.$d.L3m.DAY.CHL.chlor_a.4km.nc" >> $3
	d=`gdate --date="$d +1 day" +%Y%m%d`
done

