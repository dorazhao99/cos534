#!/bin/bash

python train.py --dataset $1 --image_path /Volumes/G-DRIVE\ mobile\ USB-C/BFW/ --labels_train ../data/$1/bfw_train.pkl --labels_val ../data/$1/bfw_val.pkl --num_epochs 1 --num_classes 4 --lr 0.0003
