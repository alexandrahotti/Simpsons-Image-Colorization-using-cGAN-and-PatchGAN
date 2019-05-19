#/bin/sh
# Crop standard defition images to 360x360
for filename in $(find -name '*.jpg'); do
   ffmpeg -y -i $filename -vf crop=iw-120:ih-0 $filename	
done


