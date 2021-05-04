import time
import os
import json
import numpy as np
import argparse
import collections
import matplotlib.pyplot as plt
from statsmodels.stats.inter_rater import fleiss_kappa

def create_histogram(eval_pred, eval_true, num_categs, category_name):
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
    category_counts = np.zeros((len(eval_pred[0]), num_categs))
    ret = {i:[] for i in range(num_categs)}
    for i,pred in enumerate(eval_pred):
        for j,p in enumerate(pred):
            category_counts[j][p[idx]] += 1
    category_counts /= len(eval_pred)
    max_inds = category_counts.argmax(axis=1)
    max_vals = category_counts.max(axis=1)
    for idx, val in zip(max_inds,max_vals):
        ret[idx].append(val)
    return ret


# example usage
# python3 code/create_figures/create_figure_2_subplot.py --evals results/fairface/bfw_on_fairface_3_eval.json results/fairface/bfw_on_fairface_2_eval.json results/fairface/bfw_on_fairface_1_eval.json --gender_to_idx data/gender.json --race_to_idx data/race.json --outdir .

parser = argparse.ArgumentParser()
parser.add_argument('--evals', nargs='+', type=str, default=[], help="list of paths to eval json files (should contain both race and gender data in form of [[gender, race],...,[gender, race]]. Can be generated from code/combine_evals.py, for laofiw just populate it with 0s/1s and ignore the output of this file for gender lol")
parser.add_argument('--gender_to_idx', type=str, required=True, help='path to gender_to_idx json, for laofiw just point to a random one')
parser.add_argument('--race_to_idx', type=str, required=True, help='path to gender_to_idx json')
parser.add_argument('--outdir', type=str, required=True, help='output directory')
arg = vars(parser.parse_args())
assert len(arg['evals']) > 0
print(arg, '\n', flush=True)

if not os.path.exists(arg['outdir']):
    os.makedirs(arg['outdir'])

gender_to_idx = json.load(open(arg['gender_to_idx']))
race_to_idx = json.load(open(arg['race_to_idx']))
print(gender_to_idx)
# Get list of all filenames
eval = json.load(open(arg['evals'][0]))

# Get all model predictions
evals_pred = []; evals_true = []
for f in arg['evals']:
    eval = json.load(open(f))
    evals_pred.append(eval['pred'])
    evals_true.append(eval['true'])

# plot gender
hist_dicts_gender = create_histogram(evals_pred, evals_true, len(gender_to_idx), category_name='gender')
fig, axs = plt.subplots(1, len(gender_to_idx), sharey=True, tight_layout=True)
fig.suptitle("Gender ")
#fig.xlabel("Label Homogeneity")
#fig.ylabel("Fraction of Individuals")
for k,v in gender_to_idx.items():
    axs[v].hist(hist_dicts_gender[v], 10, density=True)
    axs[v].set_title(k)
plt.savefig(f'{arg["outdir"]}/fig2_gender.png')

# plot race
hist_dicts_race = create_histogram(evals_pred, evals_true, len(race_to_idx), category_name='race')
fig, axs = plt.subplots(1, len(race_to_idx), sharey=True, tight_layout=True)
print(race_to_idx)
fig.suptitle("Race ")
#fig.supxlabel("Label Homogeneity")
#fig.supylabel("Fraction of Individuals")
for k,v in race_to_idx.items():
    axs[v].hist(hist_dicts_race[v], 10, density=True)
    axs[v].set_title(k)
plt.savefig(f'{arg["outdir"]}/fig2_race.png')
