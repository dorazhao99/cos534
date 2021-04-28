#!/usr/bin/env bash

filename='CC_part_19_1.txt'
n=1
lim=20

echo "Extracting frames..."
while read line; do
  echo ""
  echo "Unzipping line $n: $line"
  unzip CC_part_19_1.zip $line
  python extract_frame.py --filepath $line --threshold 75
  rm -r $line
#   if [[ $n -ge $lim ]]; then
#     break
#   fi
  ((n++))
done < $filename

echo ""
echo "Processing frames..."
python process_frames.py --threshold 75

echo ""
echo "All done!"
