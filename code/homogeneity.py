# Computes Fleiss Kappa scores for individual examples (rather than groups)
# Used later to create histograms as in Fig. 2 of the original paper

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
    print(fleiss_inputs)
    fleiss_kappas = []
    for r in range(fleiss_inputs.shape[0]):
        fleiss_kappas.append(fleiss_kappa(fleiss_inputs[r]))
    return fleiss_kappas

def create_histogram(fleiss_kappas, filename=None):
    """
    fleiss_kappas: a dictionary mapping 'humanlabel' --> array of fleiss kappa scores
    """
    fig = plt.figure(figsize=(16,4))
    for c,categ in enumerate(list(fleiss_kappas.keys())):
        ax = fig.add_subplot(1, len(fleiss_kappas), c+1)
        ax.hist(fleiss_kappas[categ], 10)
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('--evals', nargs='+', type=str, default=[])
parser.add_argument('--gender_to_idx', type=str, required=True)
parser.add_argument('--race_to_idx', type=str, required=True)
parser.add_argument('--outdir', type=str, required=True)
arg = vars(parser.parse_args())
assert len(arg['evals']) > 0
print(arg, '\n', flush=True)

if not os.path.exists(arg['outdir']):
    os.makedirs(arg['outdir'])

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

# Get all individual gender fleiss kappa scores
fleiss_kappas_all = compute_fleiss_kappas(evals_pred, evals_true, len(gender_to_idx), category_name='gender')
fleiss_kappas_by_gender = collections.defaultdict(list)
gender_true = [t for t,_ in evals_true[0]]
for i,fk in enumerate(fleiss_kappas):
    for g in gender_to_idx:
        if gender_true[i] == gender_to_idx[g]:
            fleiss_kappas_by_gender[g].append(fk)
        break

create_histogram(fleiss_kappas_by_gender, filename='{}/gender_fleiss_kappas.png'.format(arg['outdir']))
with open('{}/gender_fleiss_kappas.json'.format(arg['outdir']), 'w') as f:
    json.dump(fleiss_kappas_by_gender, f)

# Get all individual race fleiss kappa scores
fleiss_kappas_all = compute_fleiss_kappas(evals_pred, evals_true, len(race_to_idx), category_name='race')
fleiss_kappas_by_race = collections.defaultdict(list)
race_true = [t for _,t in evals_true[0]]
for i,fk in enumerate(fleiss_kappas):
    for r in race_to_idx:
        if race_true[i] == race_to_idx[r]:
            fleiss_kappas_by_race[r].append(fk)
        break

create_histogram(fleiss_kappas_by_race, filename='{}/race_fleiss_kappas.png'.format(arg['outdir']))
with open('{}/race_fleiss_kappas.json'.format(arg['outdir']), 'w') as f:
    json.dump(fleiss_kappas_by_race, f)

# Get all individual intersectional fleiss kappa scores
fleiss_kappas_all = compute_fleiss_kappas(evals_pred, evals_true, len(gender_to_idx) * len(race_to_idx))
fleiss_kappas_inter = collections.defaultdict(list)
inter_true = [len(race_to_idx) * gt + rt for gt,rt in evals_true[0]]
for i,fk in enumerate(fleiss_kappas):
    for g in gender_to_idx:
        for r in race_to_idx:
            inter_idx = len(race_to_idx) * gender_to_idx[g] + race_to_idx[r]
            if inter_true[i] == inter_idx:
                fleiss_kappas_inter['{}-{}'.format(g, r)].append(fk)
        break 

create_histogram(fleiss_kappas_inter, filename='{}/inter_fleiss_kappas.png'.format(arg['outdir']))
with open('{}/inter_fleiss_kappas.json'.format(arg['outdir']), 'w') as f:
    json.dump(fleiss_kappas_inter, f)
