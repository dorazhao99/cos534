#!/usr/bin/env bash

zip_name='CC_part_19_1.zip'
filename='CC_part_19_1.txt'
unzip -l $zip_name > $filename

python process_zip.py $filename

echo ""
echo "All done!"
