import time
import os
import json
import numpy as np
import argparse

from statsmodels.stats.inter_rater import fleiss_kappa

def compute_fleiss_kappa(pred, true, group, num_categs):
    """
    pred: a list of all predictions for race or gender
    true: a list of all ground truth labels for race or gender, in 
          the same order as the pred list
    group: an int describing the group (e.g. 0 for Female, 1 for Male, etc.) 
           to compute the Fleiss-Kappa score over
    num_categs: int for all possible category labels (e.g., 2 for gender, 4 for race)
    """

    num_subjects = true.count(group)
    fleiss_inputs = np.zeros((num_subjects, num_categs))
    idx = 0
    for i,pred in enumerate(pred):
        if true[i] == group:
            fleiss_inputs[idx][pred] += 1
            idx += 1
    return fleiss_kappa(fleiss_inputs)

parser = argparse.ArgumentParser()
parser.add_argument('--evals', nargs='+', type=str, default=[])
parser.add_argument('--gender_to_idx', type=str, required=True)
parser.add_argument('--race_to_idx', type=str, required=True)
parser.add_argument('--outfile', type=str, default='fleiss_kappas.json')
arg = vars(parser.parse_args())
assert len(arg['evals']) > 0
print(arg, '\n', flush=True)

gender_to_idx = json.load(open(arg['gender_to_idx']))
race_to_idx = json.load(open(arg['race_to_idx']))

# Get list of all filenames
eval = json.load(open(arg['evals'][0]))

# Get all model predictions
evals_pred = []
for f in arg['evals']:
    eval = json.load(open(f))
    evals_pred.append(eval['pred'])

# Compute gender fleiss kappas
gender_pred = [p for p,_ in eval['pred']]
gender_true = [t for t,_ in eval['true']]
for g in gender_to_idx:
    fleiss_kappa_score = compute_fleiss_kappa(gender_pred, gender_true, g, len(gender_to_idx))
    print('{} Fleiss-K score: {:.2f}'.format(g, fleiss_kappa_score))

# Compute race fleiss kappas
race_pred = [p for _,p in eval['pred']]
race_true = [t for _,t in eval['true']]
for r in race_to_idx:
    fleiss_kappa_score = compute_fleiss_kappa(race_pred, race_true, r, len(race_to_idx))
    print('{} Fleiss-K score: {:.2f}'.format(r, fleiss_kappa_score))

# Compute intersectional fleiss kappas
inter_pred = [10 * gp + rp for gp,rp in eval['pred']]
inter_true = [10 * gt + rt for gt,rt in eval['true']]
for g in gender_to_idx:
    for r in race_to_idx:
        inter_idx = 10 * g + r
        fleiss_kappa_score = compute_fleiss_kappa(inter_pred, inter_true, inter_idx, len(race_to_idx) * len(gender_to_idx))
        print('{}, {} Fleiss-K score: {:.2f}'.format(g, r, fleiss_kappa_score))
