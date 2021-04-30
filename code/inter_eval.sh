#!/bin/bash
#SBATCH --job-name=intereval_534
#SBATCH --output=bfw_ff_gender_intereval.txt
#
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#
#SBATCH --mail-type=all
#SBATCH --mail-user=dorothyz@princeton.edu

python inter_evaluate.py \
--modelpath_gender ../results/$1/gender/model_best.pth \
--modelpath_race ../results/$1/race/model_best.pth \
--humanlabels_gender ../data/gender.json \
--humanlabels_race ../data/race.json \
--labels_gender ../data/$2/$2_gender_test.pkl \
--labels_race ../data/$2/$2_race_test.pkl \
--batchsize 32 \
--outfile ../results/$1_$2_inter.json
