#!/bin/bash

python data_preprocess.py --labels $2.json --annotations $1/$1_annotations_fleiss.json --labels_test $1/$1_$2_test.pkl --labels_val $1/$1_$2_val.pkl \
--labels_train $1/$1_$2_train.pkl --image_path /scratch/dorothyz/$1/ --race 0
