#/bin/sh
# Rescale square images to 256x256
for filename in $(find -name '*.jpg'); do
   ffmpeg -y -i $filename -vf scale=256:256 $filename
done


