#!/bin/bash
#SBATCH --job-name=train_534
#SBATCH --output=cc_2_g.txt
#
##SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#
#SBATCH --mail-type=all
#SBATCH --mail-user=dorothyz@princeton.edu

source /home/dorothyz/.bashrc

conda activate 534

python train.py \
--labels_train $1 \
--labels_val $2 \
--num_epochs 10 \
--num_classes 2 \
--lr 0.0003 \
--outdir $3
