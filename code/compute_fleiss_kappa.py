import torch
import time
import os
import json
import numpy as np
import argparse

from statsmodels.stats.inter_rater import fleiss_kappa

parser = argparse.ArgumentParser()
parser.add_argument('--evals', nargs='+', type=str, default=[])
parser.add_argument('--outfile', type=str, default='fleiss_kappas.json')
arg = vars(parser.parse_args())
assert len(arg['evals']) > 0
print(arg, '\n', flush=True)

if not os.path.exists(arg['outdir']):
    os.makedirs(arg['outdir'])

# Get list of all filenames
eval = json.load(open(arg['evals'][0]))
filenames = eval['labels']

# Get all model predictions
evals_pred = []
for f in arg['evals']:
    eval = json.load(open(eval))
    evals_pred.append(eval['pred'])

fleiss_kappas = {} # Maps filename --> (gender_fleiss_kappa, race_fleiss_kappa)
for i,f in enumerate(filename):
    preds_gender = []; preds_race = []
    for eval in evals_pred:
        preds_gender.append(evals_pred[i][0])
        preds_race.append(evals_pred[i][1])
    fleiss_kappas[f] = [fleiss_kappa(preds_gender), fleiss_kappa(preds_race)]

with open(arg['outfile']) as f:
    json.dump(fleiss_kappas, f)
