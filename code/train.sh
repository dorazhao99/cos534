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

source /home/dorothyz/.bashrc

conda activate 534

python train.py \
--labels_train ../data/LAOFIW/laofiw_train.pkl \
--labels_val ../data/LAOFIW/laofiw_val.pkl \
--num_epochs 15 \
--num_classes 4 \
--outdir ../results/LAOFIW/

