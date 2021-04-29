#!/bin/bash
#SBATCH --job-name=train_534
#SBATCH --output=bfw_total.txt
#
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#
#SBATCH --mail-type=all
#SBATCH --mail-user=dorothyz@princeton.edu

python train.py --dataset $1 \
--labels_train ../data/$1/bfw_race_train.pkl \
--labels_val ../data/$1/bfw_race_val.pkl \
--num_epochs 25 \
--num_classes 4 \
--lr 0.0003 \
--outdir ../results/$1_total
