#/bin/sh
# To convert an image to gray scale 
for filename in $(find -name '*.jpg'); do
	file="${filename:0:-4}"
	gray="_gray.jpg"
	outname="$file$gray"
	convert $filename -colorspace Gray $outname
done

