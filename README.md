# COS 534
Final Project for COS 534: Fairness in Machine Learning

## Documentation

#### Folder `/data` notes

*Casual Conversations*

Make sure you have this file in the `data/` folder: 

https://github.com/tzutalin/dlib-android/blob/master/data/shape_predictor_68_face_landmarks.dat

How to process:

1. Download the appropriate .zip file from https://ai.facebook.com/datasets/casual-conversations-dataset/ and rename to something like `CC_part_1_1.zip`.

2. Run `./get_zip_files.sh <ZIP_FILENAME> <OUT_TXT_NAME>`. Here, `ZIP_FILENAME` is the name of the downloaded .zip file, and `OUT_TXT_NAME` is the name of the output .txt file where the .zip filenames are printed to.

3. Run `./process_videos.sh <ZIP_FILENAME> <OUT_TXT_NAME>`. Use the same names as above.

*FairFace*

Notes on data formatting:

- Labels map full img path names to (0, 1) for ("Male, "Female") or (0, 1, 2, 3) for ("White", "Black", "Asian", "Indian"). "East Asian" and "Southeast Asian" labels were grouped together into the "Asian" category, and "Middle Eastern" and "Hispanic\_Latino" images were not processed (as per the original paper)

- Train/Val split was obtained by doing an 80-20 split on the train set.

## Training

**NOTE:** Make sure label mappings are consistent across datasets! Use the mapping in `data/gender.json` and `data/race.json`.

All pre-trained models are backed up on the shared Google drive. Try not to add too many models to the GitHub repo because they take up a lot of storage.

*FairFace*

- Models were trained for 5 epochs, using Adam optimizer and learning rate of 1e-4. The model with the highest validation accuracy was saved.

## Evaluation

- `evaluate.py`: Used to evaluate a model trained on one dataset on any other test dataset; evaluates only gender or only race.
- `inter_evaluate.py`: Used to evaluate a model trained on one dataset on any other test dataset; does gender-race pairs.

## Analysis

*Computing Fleiss Kappa scores*

`compute_fleiss_kappas.py`

The argument `--evals` should be a list of all .json files that have prediction and ground truths. For example, if we are measuring agreement between LAOFIW, BFW, CC, FairFace on race, then it should be a list of four .json files containing the output race predictions and ground truths (and filepaths to the images).

The arguments `--race_to_idx` and `--gender_to_idx` should point to the .json files containing the humanlabel to index mapping for possible labels (e.g. 'F': 0, 'M': 1 for the case of gender). If only measuring agreement between race (i.e. anytime LAOFIW is included), then set `--gender_to_idx` to `None` (i.e. just do not include the flag in the shell script, it is set to `None` by default). If both `--race_to_idx` and `--gender_to_idx` are set, then an intersectional analysis will also be conducted; otherwise, only one or the other will be analyzed.

Sample usages:

```
EVALS=("fairface_eval.json" "bfw_eval.json" "laofiw_eval.json")
RACE_TO_IDX="race.json"

python compute_fleiss_kappa.py --evals ${EVALS[*]} --race_to_idx $RACE_TO_IDX
```


```
EVALS=("fairface_eval.json" "bfw_eval.json" "cc_eval.json")
RACE_TO_IDX="race.json"
GENDER_TO_IDX="gender.json"

python compute_fleiss_kappa.py --evals ${EVALS[*]} --race_to_idx $RACE_TO_IDX --gender_to_idx $GENDER_TO_IDX
```
