# COS 534
Final Project for COS 534: Fairness in Machine Learning

## Documentation

#### Folder `/data` notes

*Casual Conversations*

How to process:

1. Download the appropriate .zip file from https://ai.facebook.com/datasets/casual-conversations-dataset/ and rename to something like `CC_part_1_1.zip`.

2. In `get_zip_files.sh`, set `zip_name` to the same filename in Step 1, and set `filename` to something like `CC_part_1_1.txt` (make sure it is a .txt file).

3. Run `./get_zip_files.sh`.

4. In `process_videos.sh`, set `filename` to the same .txt filename in Step 2. If you only want to process some of the videos, set `lim` to the line in the .txt file that you want to stop at and uncomment lines 14-16.

5. Run `./process_videos.sh`.

*FairFace*

Notes on data formatting:

- Labels map full img path names to (0, 1) for ("Male, "Female") or (0, 1, 2, 3) for ("White", "Black", "Asian", "Indian"). "East Asian" and "Southeast Asian" labels were grouped together into the "Asian" category, and "Middle Eastern" and "Hispanic\_Latino" images were not processed (as per the original paper)

- Train/Val split was obtained by doing an 80-20 split on the train set.

## Training

**NOTE:** Make sure label mappings are consistent across datasets! Use the mapping in `data/gender.json` and `data/race.json`.

*FairFace*

Models were trained for 5 epochs, using Adam optimizer and learning rate of 1e-4.
