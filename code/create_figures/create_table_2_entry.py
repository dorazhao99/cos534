import time
import os
import json
import numpy as np
import argparse
import collections

from statsmodels.stats.inter_rater import fleiss_kappa

def compute_fleiss_kappas(eval_pred, eval_true, num_categs, category_name):
    """
    eval_pred: a list of lists of all predictions for race or gender for each model
    eval_true: a list of lists of all ground truth labels for race or gender, in 
          the same order as the pred list, for each model
    num_categs: int for all possible category labels (e.g., 2 for gender, 4 for race)
    """
    idx = -1
    if category_name=='gender':
        idx = 0
    elif category_name == 'race':
        idx = 1
    fleiss_inputs = np.zeros((len(eval_pred[0]), num_categs))
    for i,pred in enumerate(eval_pred):
        for j,p in enumerate(pred):
            fleiss_inputs[j][p[idx]] += 1
    return fleiss_kappa(fleiss_inputs)

parser = argparse.ArgumentParser()
parser.add_argument('--evals', nargs='+', type=str, default=[])
parser.add_argument('--gender_to_idx', type=str, required=True)
parser.add_argument('--race_to_idx', type=str, required=True)
arg = vars(parser.parse_args())
assert len(arg['evals']) > 0
print(arg, '\n', flush=True)

gender_to_idx = json.load(open(arg['gender_to_idx']))
race_to_idx = json.load(open(arg['race_to_idx']))

# Get list of all filenames
eval = json.load(open(arg['evals'][0]))

# Get all model predictions
evals_pred = []; evals_true = []
for f in arg['evals']:
    eval = json.load(open(f))
    evals_pred.append(eval['pred'])
    evals_true.append(eval['true'])

fleiss_kappa_gender = compute_fleiss_kappas(evals_pred, evals_true, len(gender_to_idx), category_name='gender')
fleiss_kappa_race = compute_fleiss_kappas(evals_pred, evals_true, len(race_to_idx), category_name='race')
print(f"""
gender - {fleiss_kappa_gender}
race - {fleiss_kappa_race}
""")
