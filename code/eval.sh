#!/bin/bash
#SBATCH --job-name=eval_534
#SBATCH --output=bfw_gender_eval.txt
#
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#
#SBATCH --mail-type=all
#SBATCH --mail-user=dorothyz@princeton.edu

python evaluate.py \
--humanlabels $3 \
--modelpath $1 \
--labels_test $2 \
--batchsize 32 \
--num_classes 4 \
--outfile $4
