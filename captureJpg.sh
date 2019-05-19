#/bin/sh
# Capture one frame every 15th second
for filename in $(find -name "*.mp4"); do
    ffmpeg -i "$filename" -vf fps=4/60 -ss 00:01:00 -to 00:19:00  "$filename%04d.jpg"

done
