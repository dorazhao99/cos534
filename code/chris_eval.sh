#!/bin/bash
#SBATCH --job-name=eval_534
#SBATCH --output=outfiles/$1_$2_on_laofiw_eval.txt
#
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#
#SBATCH --mail-type=all
#SBATCH --mail-user=czye@princeton.edu

python evaluate.py \
--humanlabels ../data/race.json \
--modelpath /scratch/network/czye/race_models/$1_best_$2.pth \
--labels_test ../data/LAOFIW/laofiw_test.pkl \
--batchsize 32 \
--num_classes 4 \
--outfile ../results/LAOFIW/val/$1_$2_on_laofiw.json
