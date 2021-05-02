# COS 534
Final Project for COS 534: Fairness in Machine Learning

## Documentation

#### Folder `/data` notes

*Casual Conversations*

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

*FairFace*

- Models were trained for 5 epochs, using Adam optimizer and learning rate of 1e-4. The model with the highest validation accuracy was saved.

## Evaluation

- `evaluate.py`: Used to evaluate a model trained on one dataset on any other test dataset; evaluates only gender or only race.
- `inter_evaluate.py`: Used to evaluate a model trained on one dataset on any other test dataset; does gender-race pairs.

## Analysis

*Computing Fleiss Kappa scores*

The method `fleiss_kappa` from `statsmodel` is used. The Fleiss-Kappa score is evaluated for each group, e.g. 'Black', 'Female', 'Asian-Male', etc. The input consists of a `num_subjects * num_categories` table, where row r of this table corresponds to a particular input image with ground truth label as the group being evaluated. 

Each column corresponds to a possible label (e.g. if we are evaluating 'Male', then there are two possible labels, 'Male' and 'Female'). To fill in the table, tally up all predictions assigned by each dataset model on each example. Thus the sum over each row should be identical, and should equal the number of models tested.
