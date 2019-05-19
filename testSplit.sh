#/bin/sh
# Move 1000 files to the test set
for filename in $(find -name '*.jpg' | shuf -n 1000); do
	mv $filename testset/
done

