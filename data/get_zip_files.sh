#!/usr/bin/env bash

# $1 is the name of the .zip file
# $2 is the name of the output .txt file

unzip -l $1 > $2

python process_zip.py $2

echo ""
echo "All done!"
