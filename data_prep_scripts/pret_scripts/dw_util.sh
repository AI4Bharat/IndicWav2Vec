
nt=$3 #number of threads


fname=$(basename "$1")

folder=${fname[@]:0:-4}

root=$2
save_dir="$root/raw_data/$folder"


if [ -d $save_dir ] 
then mkdir -p $save_dir 
fi

cat $1 | xargs -I '{}' -P $nt yt-dlp -f "bestaudio/best" -ciw -o "$save_dir/%(id)s.%(ext)s" --extract-audio --audio-format wav --audio-quality 0 --no-playlist {} --ppa "ffmpeg:-ac 1 -ar 16000" #--quiet &&

python vad.py "$root/raw_data/" "$root/data_refined/" $folder &&
python snr_filter.py "$root/data_refined/" $folder "$root/snr_rejected/"&&
python chunking.py "$root/data_refined/"$folder 


