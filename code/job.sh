#!/bin/sh
#SBATCH --job-name=train_534
#
#SBATCH -N 1                   # number of nodes requested
#SBATCH -n 1                   # number of tasks requested
#SBATCH --ntasks-per-node 1    # number of tasks per node
#
#SBATCH --mem-per-cpu=4G       # memory per cpu-core (4G default)
#SBATCH --gres=gpu:1           # number of GPUs requested
#SBATCH -t 20:00:00            # time requested in hour:minute:second
#
#SBATCH --mail-type=all
#SBATCH --mail-user=sharonz@princeton.edu

source /n/fs/interp-scr/sharonz/interp/bin/activate

JOB=11

if [[ $JOB == 0 ]]
then
  echo Training classifier

  TYPE="gender"
  DATASET="fairface"
  LABELS_TRAIN="/n/fs/visualai-scr/sharonz/fairface/labels_${TYPE}_train_80.pkl"
  LABELS_VAL="/n/fs/visualai-scr/sharonz/fairface/labels_${TYPE}_train_20.pkl"
  NUM_EPOCHS=15
  if [[ $TYPE == "gender" ]]
  then
    NUM_CLASSES=2
  else
    NUM_CLASSES=4
  fi
  LR=0.0001
  BATCHSIZE=64
  PRINT_FREQ=500
  OUTDIR="../results/${DATASET}/${TYPE}"

  python train.py \
    --labels_train $LABELS_TRAIN --labels_val $LABELS_VAL \
    --batchsize $BATCHSIZE --num_epochs $NUM_EPOCHS --num_classes $NUM_CLASSES --lr $LR \
    --print_freq $PRINT_FREQ --outdir $OUTDIR
fi

if [[ $JOB == 1 ]]
then
  echo Evaluating classifier

  TYPE="gender"
  MODELPATH="/n/fs/visualai-scr/sharonz/cos534/results/fairface/${TYPE}/model_best.pth"
  OUTFILE="/n/fs/visualai-scr/sharonz/cos534/results/fairface/${TYPE}/fairface_eval.json"
  LABELS_TEST="/n/fs/visualai-scr/sharonz/fairface/labels_${TYPE}_val.pkl"
  HUMANLABELS="/n/fs/visualai-scr/sharonz/cos534/data/${TYPE}.json"
  BATCHSIZE=64
  if [[ $TYPE == "gender" ]]
  then
    NUM_CLASSES=2
  else
    NUM_CLASSES=4
  fi

  python evaluate.py \
    --modelpath $MODELPATH --labels_test $LABELS_TEST --humanlabels $HUMANLABELS \
    --batchsize $BATCHSIZE --num_classes $NUM_CLASSES --outfile $OUTFILE
fi

if [[ $JOB == 2 ]]
then
  echo Evaluating intersectional classifiers

  FOLDER="/n/fs/visualai-scr/sharonz/cos534"
  MODELPATH_RACE="${FOLDER}/results/fairface/race/model_best.pth"
  MODELPATH_GENDER="${FOLDER}/results/fairface/gender/model_best.pth"
  LABELS_RACE="/n/fs/visualai-scr/sharonz/fairface/labels_race_val.pkl"
  LABELS_GENDER="/n/fs/visualai-scr/sharonz/fairface/labels_gender_val.pkl"
  HUMANLABELS_RACE="${FOLDER}/data/race.json"
  HUMANLABELS_GENDER="${FOLDER}/data/gender.json"
  BATCHSIZE=64
  NUM_CLASSES_RACE=4
  NUM_CLASSES_GENDER=2

  python inter_evaluate.py \
    --modelpath_race $MODELPATH_RACE --labels_race $LABELS_RACE --humanlabels_race $HUMANLABELS_RACE --num_classes_race $NUM_CLASSES_RACE \
    --modelpath_gender $MODELPATH_GENDER --labels_gender $LABELS_GENDER --humanlabels_gender $HUMANLABELS_GENDER --num_classes_gender $NUM_CLASSES_GENDER\
    --batchsize $BATCHSIZE
fi

if [[ $JOB == 3 ]]
then
  echo Cross-evaluating classifiers on FairFace dataset

  TYPE="gender"
  DATASET="bfw"
  FOLDER="/n/fs/visualai-scr/sharonz/cos534"
  MODELPATH="${FOLDER}/results/${DATASET}/${TYPE}/model_best.pth"
  OUTFILE="${FOLDER}/results/fairface/${TYPE}/${DATASET}_eval.json"
  LABELS_TEST="/n/fs/visualai-scr/sharonz/fairface/labels_${TYPE}_val.pkl"
  HUMANLABELS="${FOLDER}/data/${TYPE}.json"
  BATCHSIZE=64
  if [[ $TYPE == "gender" ]]
  then
    NUM_CLASSES=2
  else
    NUM_CLASSES=4
  fi

  python evaluate.py \
    --modelpath $MODELPATH --labels_test $LABELS_TEST --humanlabels $HUMANLABELS \
    --batchsize $BATCHSIZE --num_classes $NUM_CLASSES --outfile $OUTFILE
fi

if [[ $JOB == 4 ]]
then
  echo Evaluating intersectional classifiers on FairFace dataset

  DATASET="bfw"
  FOLDER="/n/fs/visualai-scr/sharonz/cos534"
  MODELPATH_RACE="${FOLDER}/results/${DATASET}/race/model_best.pth"
  MODELPATH_GENDER="${FOLDER}/results/${DATASET}/gender/model_best.pth"
  LABELS_RACE="/n/fs/visualai-scr/sharonz/fairface/labels_race_val.pkl"
  LABELS_GENDER="/n/fs/visualai-scr/sharonz/fairface/labels_gender_val.pkl"
  HUMANLABELS_RACE="${FOLDER}/data/race.json"
  HUMANLABELS_GENDER="${FOLDER}/data/gender.json"
  BATCHSIZE=64
  NUM_CLASSES_RACE=4
  NUM_CLASSES_GENDER=2

  python inter_evaluate.py \
    --modelpath_race $MODELPATH_RACE --labels_race $LABELS_RACE --humanlabels_race $HUMANLABELS_RACE --num_classes_race $NUM_CLASSES_RACE \
    --modelpath_gender $MODELPATH_GENDER --labels_gender $LABELS_GENDER --humanlabels_gender $HUMANLABELS_GENDER --num_classes_gender $NUM_CLASSES_GENDER\
    --batchsize $BATCHSIZE
fi

if [[ $JOB == 5 ]]
then
  echo Combine race and gender predictions and ground truths

  DATASET_1="bfw"
  DATASET_2="fairface"

  RACE_EVALS="/n/fs/visualai-scr/sharonz/cos534/results/${DATASET_2}/race/${DATASET_1}_eval.json"
  GENDER_EVALS="/n/fs/visualai-scr/sharonz/cos534/results/${DATASET_2}/gender/${DATASET_1}_eval.json"
  OUTFILE="/n/fs/visualai-scr/sharonz/cos534/results/${DATASET_2}/${DATASET_1}_on_${DATASET_2}_eval.json"

  python combine_evals.py --race_evals $RACE_EVALS --gender_evals $GENDER_EVALS --outfile $OUTFILE
fi

if [[ $JOB == 6 ]]
then
  echo Compute Fleiss Kappa scores for Table 3

  FOLDER="/n/fs/visualai-scr/sharonz/cos534"
  EVALS=("${FOLDER}/results/fairface/fairface_on_fairface_eval.json" \
         "${FOLDER}/results/fairface/bfw_on_fairface_eval.json")
  RACE_TO_IDX="${FOLDER}/data/race.json"
  GENDER_TO_IDX="${FOLDER}/data/gender.json"

  python compute_fleiss_kappa.py --evals ${EVALS[*]} --race_to_idx $RACE_TO_IDX --gender_to_idx $GENDER_TO_IDX
fi

if [[ $JOB == 7 ]]
then
  echo Compute individual Fleiss Kappa scores and create homogeneity plots

  FOLDER="/n/fs/visualai-scr/sharonz/cos534"
  EVALS=("${FOLDER}/results/fairface/fairface_on_fairface_eval.json" \
         "${FOLDER}/results/fairface/bfw_on_fairface_eval.json")
  RACE_TO_IDX="${FOLDER}/data/race.json"
  GENDER_TO_IDX="${FOLDER}/data/gender.json"
  OUTDIR="${FOLDER}/results/fairface"

  python homogeneity.py --evals ${EVALS[*]} --race_to_idx $RACE_TO_IDX --gender_to_idx $GENDER_TO_IDX --outdir $OUTDIR
fi

if [[ $JOB == 8 ]]
then
  echo Training three-way split ensemble
  
  TYPE="race"
  DATASET="fairface"
  SPLIT="1"
  LABELS_TRAIN="/n/fs/visualai-scr/sharonz/fairface/labels_${TYPE}_train_${SPLIT}.pkl"
  LABELS_VAL="/n/fs/visualai-scr/sharonz/fairface/labels_${TYPE}_train_20.pkl"
  NUM_EPOCHS=5
  if [[ $TYPE == "gender" ]]
  then
    NUM_CLASSES=2
  else
    NUM_CLASSES=4
  fi
  LR=0.0001
  BATCHSIZE=64
  PRINT_FREQ=500
  OUTDIR="../results/${DATASET}/${TYPE}_${SPLIT}"

  python train.py \
    --labels_train $LABELS_TRAIN --labels_val $LABELS_VAL \
    --batchsize $BATCHSIZE --num_epochs $NUM_EPOCHS --num_classes $NUM_CLASSES --lr $LR \
    --print_freq $PRINT_FREQ --outdir $OUTDIR
fi

if [[ $JOB == 9 ]]
then
  echo Evaluating three-way ensembles on FairFace

  TYPE="gender"
  SPLIT="1"
  DATASET="bfw"
  FOLDER="/n/fs/visualai-scr/sharonz/cos534"
  MODELPATH="/n/fs/visualai-scr/sharonz/other/${DATASET}/${TYPE}_${SPLIT}/model_best.pth"
  OUTFILE="${FOLDER}/results/fairface/${TYPE}_${SPLIT}/${DATASET}_eval.json"
  LABELS_TEST="/n/fs/visualai-scr/sharonz/fairface/labels_${TYPE}_val.pkl"
  HUMANLABELS="${FOLDER}/data/${TYPE}.json"
  BATCHSIZE=64
  if [[ $TYPE == "gender" ]]
  then
    NUM_CLASSES=2
  else
    NUM_CLASSES=4
  fi

  python evaluate.py \
    --modelpath $MODELPATH --labels_test $LABELS_TEST --humanlabels $HUMANLABELS \
    --batchsize $BATCHSIZE --num_classes $NUM_CLASSES --outfile $OUTFILE
fi

if [[ $JOB == 10 ]]
then
  echo Combine race and gender predictions and ground truths

  DATASET_1="bfw"
  DATASET_2="fairface"
  SPLIT="1"

  RACE_EVALS="/n/fs/visualai-scr/sharonz/cos534/results/${DATASET_2}/race_${SPLIT}/${DATASET_1}_eval.json"
  GENDER_EVALS="/n/fs/visualai-scr/sharonz/cos534/results/${DATASET_2}/gender_${SPLIT}/${DATASET_1}_eval.json"
  OUTFILE="/n/fs/visualai-scr/sharonz/cos534/results/${DATASET_2}/${DATASET_1}_on_${DATASET_2}_${SPLIT}_eval.json"

  python combine_evals.py --race_evals $RACE_EVALS --gender_evals $GENDER_EVALS --outfile $OUTFILE
fi

if [[ $JOB == 11 ]]
then
  echo Compute Fleiss Kappa scores for Table 2

  DATASET_1="bfw"
  DATASET_2="fairface"

  FOLDER="/n/fs/visualai-scr/sharonz/cos534"
  EVALS=("${FOLDER}/results/${DATASET_2}/${DATASET_1}_on_${DATASET_2}_1_eval.json" \
         "${FOLDER}/results/${DATASET_2}/${DATASET_1}_on_${DATASET_2}_2_eval.json" \
         "${FOLDER}/results/${DATASET_2}/${DATASET_1}_on_${DATASET_2}_3_eval.json")
  RACE_TO_IDX="${FOLDER}/data/race.json"
  GENDER_TO_IDX="${FOLDER}/data/gender.json"

  python compute_fleiss_kappa.py --evals ${EVALS[*]} --race_to_idx $RACE_TO_IDX --gender_to_idx $GENDER_TO_IDX
fi
