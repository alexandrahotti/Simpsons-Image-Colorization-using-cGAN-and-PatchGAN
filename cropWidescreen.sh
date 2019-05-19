#/bin/sh
# Crop widescreen images to square
for filename in $(find -name '*.jpg'); do
   ffmpeg -y -i $filename -vf crop=iw-280:ih-0 $filename	
done


