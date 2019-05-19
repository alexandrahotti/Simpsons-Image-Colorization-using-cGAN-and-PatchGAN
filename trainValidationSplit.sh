#/bin/sh
# Move 7000 images to the validation set
for filename in $(find -name '*.jpg' | shuf -n 7000); do
	echo $filename
done

