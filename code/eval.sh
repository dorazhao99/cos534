#!/bin/bash
#SBATCH --job-name=eval_534
#SBATCH --output=bfw_race_eval.txt
#
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#
#SBATCH --mail-type=all
#SBATCH --mail-user=dorothyz@princeton.edu

python evaluate.py \
--do_gender 0 \
--modelpath ../results/$1_$2/model_best.pth \
--labels_test ../data/$1/$1_$2_test.pkl \
--batchsize 32 
