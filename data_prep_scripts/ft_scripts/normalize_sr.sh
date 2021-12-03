path="$1"
ext=".$2"
for f in $(find "$path" -type f -name "*$ext")
do
ffmpeg -loglevel warning -hide_banner -stats -i "$f" -ar 16000 -ac 1 "$f$ext" && rm "$f" && mv "$f$ext" "$f" &

done

