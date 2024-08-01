bash write_dates.sh "$1" "$2" "$3"

python3 obdaac_download.py -v --filelist "$3" --odir "$4" 
