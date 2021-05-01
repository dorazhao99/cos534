#!/usr/bin/env bash

# $1 is the name of the .zip file
# $2 is the name of the .txt file

echo "Extracting frames..."
while read line; do
  echo ""
  echo "Unzipping line $n: $line"
  unzip $1 $line
  python extract_frame.py --filepath $line
  rm -r $line
  ((n++))
done < $2

echo ""
echo "Processing frames..."
python process_frames.py

echo ""
echo "All done!"
