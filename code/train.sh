#!/bin/bash
#SBATCH --job-name=train_534_laofiw
#SBATCH --output=laofiw_total.txt
#
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#
#SBATCH --mail-type=all
#SBATCH --mail-user=czye@princeton.edu

python train.py \
--labels_train ../data/LAOFIW/train.pkl \
--labels_val ../data/LAOFIW/val.pkl \
--num_epochs 15 \
--num_classes 4 \
--outdir ../results/LAOFIW/

