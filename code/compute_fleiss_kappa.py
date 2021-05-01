import time
import os
import json
import numpy as np
import argparse

from statsmodels.stats.inter_rater import fleiss_kappa

def compute_fleiss_kappa(eval_pred, eval_true, group, num_categs):
    """
    eval_pred: a list of lists of all predictions for race or gender for each model
    eval_true: a list of lists of all ground truth labels for race or gender, in 
          the same order as the pred list, for each model
    group: an int describing the group (e.g. 0 for Female, 1 for Male, etc.) 
           to compute the Fleiss-Kappa score over
    num_categs: int for all possible category labels (e.g., 2 for gender, 4 for race)
    """
    num_subjects = len(np.where(np.array(eval_true)[0]==group)[0])
    fleiss_inputs = np.zeros((num_subjects, num_categs))
    for i,pred in enumerate(eval_pred):
        idx = 0
        for j,p in enumerate(pred):
            if eval_true[i][j] == group:
                fleiss_inputs[idx][p] += 1
                idx += 1
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

# Compute gender fleiss kappas
gender_pred = [[p for p,_ in eval] for eval in evals_pred]
gender_true = [[t for t,_ in eval] for eval in evals_true]
for g in gender_to_idx:
    fleiss_kappa_score = compute_fleiss_kappa(gender_pred, gender_true, gender_to_idx[g], len(gender_to_idx))
    print('{} Fleiss-K score: {:.2f}'.format(g, fleiss_kappa_score))

# Compute race fleiss kappas
race_pred = [[p for _,p in eval] for eval in evals_pred]
race_true = [[t for _,t in eval] for eval in evals_true]
for r in race_to_idx:
    fleiss_kappa_score = compute_fleiss_kappa(race_pred, race_true, race_to_idx[r], len(race_to_idx))
    print('{} Fleiss-K score: {:.2f}'.format(r, fleiss_kappa_score))

# Compute intersectional fleiss kappas
inter_pred = [[len(race_to_idx) * gp + rp for gp,rp in eval] for eval in evals_pred]
inter_true = [[len(race_to_idx) * gt + rt for gt,rt in eval] for eval in evals_true]
for g in gender_to_idx:
    for r in race_to_idx:
        inter_idx = len(race_to_idx) * gender_to_idx[g] + race_to_idx[r]
        fleiss_kappa_score = compute_fleiss_kappa(inter_pred, inter_true, inter_idx, len(race_to_idx) * len(gender_to_idx))
        print('{}, {} Fleiss-K score: {:.2f}'.format(g, r, fleiss_kappa_score))