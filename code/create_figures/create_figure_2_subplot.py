import time
import os
import json
import numpy as np
import argparse
import collections
import matplotlib.pyplot as plt
from statsmodels.stats.inter_rater import fleiss_kappa
import csv

def create_histogram_fairface(eval_pred, eval_true, num_categs, category_name):
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
    category_counts = (category_counts.T / (np.sum(category_counts,axis=1) + 1e-8)).T
    max_inds = category_counts.argmax(axis=1)
    max_vals = category_counts.max(axis=1)
    for idx, val in zip(max_inds,max_vals):
        ret[idx].append(val)
    return ret

def create_histogram(eval_pred, eval_true, num_categs):
    """
    eval_pred: a list of lists of all predictions for race or gender for each model
    eval_true: a list of lists of all ground truth labels for race or gender, in 
          the same order as the pred list, for each model
    num_categs: int for all possible category labels (e.g., 2 for gender, 4 for race)
    """
    category_counts = np.zeros((len(eval_pred[0]), num_categs))
    ret = {i:[] for i in range(num_categs)}
    for i,pred in enumerate(eval_pred):
        for j,p in enumerate(pred):
            category_counts[j][p] += 1
    category_counts = (category_counts.T / (np.sum(category_counts,axis=1) + 1e-8)).T
    max_inds = category_counts.argmax(axis=1)
    max_vals = category_counts.max(axis=1)
    for idx, val in zip(max_inds,max_vals):
        ret[idx].append(val)
    return ret

def make_json(csvFilePath):
     
    # create a dictionary
    data = {}
    ids = set()
    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
         
        # Convert each row into a dictionary
        # and add it to data
        for rows in csvReader:
             
            # Assuming a column named 'No' to
            # be the primary key
            key = rows['p1']
            data[f'../../bfw/{key}'] = rows['id1']
            ids.add(rows['id1'])
    return data, len(ids)

def create_histogram_bfw(eval_pred, eval_true, labels, file_name_to_id, num_individuals, num_categs, category_name):
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
    category_counts = np.zeros((num_individuals, num_categs))
    ret = {i:[] for i in range(num_categs)}
    for i, (pred, label) in enumerate(zip(eval_pred,labels)):
        for j,(p,l) in enumerate(zip(pred, label)):
            individual = int(file_name_to_id[l])
            category_counts[individual][p[idx]] += 1
    category_counts = (category_counts.T / (np.sum(category_counts,axis=1) + 1e-8)).T
    max_inds = category_counts.argmax(axis=1)
    max_vals = category_counts.max(axis=1)
    for idx, val in zip(max_inds,max_vals):
        ret[idx].append(val)
    return ret
def bwf_get_histogram(file_paths,ann_path,gender_to_idx,race_to_idx):
# file_paths = ["/Users/sahanpaliskara/Documents/cos534/results/bfw/bfw_on_bfw_eval.json", "/Users/sahanpaliskara/Documents/cos534/results/bfw/fairface_on_bfw_eval.json"]
# ann_path = "/Users/sahanpaliskara/Documents/bfw-v0.1.5-datatable.csv"
# gender_path = "/Users/sahanpaliskara/Documents/cos534/data/gender.json"
# race_path = "/Users/sahanpaliskara/Documents/cos534/data/race.json"

    # Get list of all filenames
    evals_pred = []; evals_true = []; labels = []
    for f in file_paths:
        eval = json.load(open(f))
        evals_pred.append(eval['pred'])
        evals_true.append(eval['true'])
        labels.append(eval['labels'])
    file_name_to_id, num_individuals = make_json(ann_path)
    hist_dicts_gender = create_histogram_bfw(evals_pred, evals_true, labels, file_name_to_id, num_individuals, len(gender_to_idx), category_name='gender')
    hist_dicts_race = create_histogram_bfw(evals_pred, evals_true, labels, file_name_to_id, num_individuals, len(race_to_idx), category_name='race')
    return hist_dicts_gender, hist_dicts_race
def combine_histograms(hists, cat_to_idx):
    cats = len(cat_to_idx)
    ret = {}
    for i in range(cats):
        all_labels = []
        for hist in hists:
            if i in hist:
                all_labels.append(hist[i])
        ret[i] = np.concatenate(all_labels)
    return ret

def get_histogram(file_paths,cat_to_idx):

    # Get list of all filenames
    evals_pred = []; evals_true = []
    for f in file_paths:
        eval = json.load(open(f))
        evals_pred.append(eval['pred'])
        evals_true.append(eval['true'])
    hist_dicts = create_histogram(evals_pred, evals_true, len(cat_to_idx))
    return hist_dicts

def fairface_get_histogram(file_paths,gender_to_idx,race_to_idx):
# file_paths = ["/Users/sahanpaliskara/Documents/cos534/results/bfw/bfw_on_bfw_eval.json", "/Users/sahanpaliskara/Documents/cos534/results/bfw/fairface_on_bfw_eval.json"]
# ann_path = "/Users/sahanpaliskara/Documents/bfw-v0.1.5-datatable.csv"
# gender_path = "/Users/sahanpaliskara/Documents/cos534/data/gender.json"
# race_path = "/Users/sahanpaliskara/Documents/cos534/data/race.json"

    # Get list of all filenames
    evals_pred = []; evals_true = []
    for f in file_paths:
        eval = json.load(open(f))
        evals_pred.append(eval['pred'])
        evals_true.append(eval['true'])
        # labels.append(eval['labels'])
    # file_name_to_id, num_individuals = make_json(ann_path)
    hist_dicts_gender = create_histogram_fairface(evals_pred, evals_true, len(gender_to_idx), category_name='gender')
    hist_dicts_race = create_histogram_fairface(evals_pred, evals_true, len(race_to_idx), category_name='race')
    return hist_dicts_gender, hist_dicts_race


# example usage
# python3 code/create_figures/create_figure_2_subplot.py --evals results/fairface/bfw_on_fairface_3_eval.json results/fairface/bfw_on_fairface_2_eval.json results/fairface/bfw_on_fairface_1_eval.json --gender_to_idx data/gender.json --race_to_idx data/race.json --outdir .

parser = argparse.ArgumentParser()
parser.add_argument('--evals_bfw', nargs='+', type=str, default=[], help="list of paths to eval json files (should contain both race and gender data in form of [[gender, race],...,[gender, race]]. Can be generated from code/combine_evals.py, for laofiw just populate it with 0s/1s and ignore the output of this file for gender lol")
parser.add_argument('--evals_laofiw', nargs='+', type=str, default=[], help="list of paths to eval json files (should contain both race and gender data in form of [[gender, race],...,[gender, race]]. Can be generated from code/combine_evals.py, for laofiw just populate it with 0s/1s and ignore the output of this file for gender lol")
parser.add_argument('--evals_cc', nargs='+', type=str, default=[], help="list of paths to eval json files (should contain both race and gender data in form of [[gender, race],...,[gender, race]]. Can be generated from code/combine_evals.py, for laofiw just populate it with 0s/1s and ignore the output of this file for gender lol")
parser.add_argument('--evals_fairface', nargs='+', type=str, default=[], help="list of paths to eval json files (should contain both race and gender data in form of [[gender, race],...,[gender, race]]. Can be generated from code/combine_evals.py, for laofiw just populate it with 0s/1s and ignore the output of this file for gender lol")
parser.add_argument('--gender_to_idx', type=str, required=True, help='path to gender_to_idx json, for laofiw just point to a random one')
parser.add_argument('--race_to_idx', type=str, required=True, help='path to gender_to_idx json')
parser.add_argument('--outdir', type=str, required=True, help='output directory')
parser.add_argument('--bfw_ann_path', type=str, required=True, help='ann path for bfw')
arg = vars(parser.parse_args())
# assert len(arg['evals']) > 0
print(arg, '\n', flush=True)

gender_to_idx = json.load(open(arg['gender_to_idx']))
race_to_idx = json.load(open(arg['race_to_idx']))
print(gender_to_idx)
# Get list of all filenames
# eval = json.load(open(arg['evals'][0]))

# Get all model predictions

hist_dicts_gender_bfw, hist_dicts_race_bfw = bwf_get_histogram(arg['evals_bfw'],arg['bfw_ann_path'],gender_to_idx,race_to_idx)
hist_dicts_gender_fairface, hist_dicts_race_fairface = fairface_get_histogram(arg['evals_fairface'],gender_to_idx,race_to_idx)
hist_dicts_gender_cc = get_histogram(arg['evals_cc'],gender_to_idx)
hist_dicts_race_laofiw = get_histogram(arg['evals_laofiw'],race_to_idx)

hist_dicts_race = combine_histograms([hist_dicts_race_bfw,hist_dicts_race_fairface,hist_dicts_race_laofiw], race_to_idx)
hist_dicts_gender = combine_histograms([hist_dicts_gender_bfw, hist_dicts_gender_fairface, hist_dicts_gender_cc], gender_to_idx)
# plot gender
fig, axs = plt.subplots(1, len(gender_to_idx), sharey=True, tight_layout=True)
fig.suptitle("Gender ")
fig.supxlabel("Label Homogeneity")
fig.supylabel("Fraction of Individuals")

colors = ["blue","orange","green","red"]

for k,v in gender_to_idx.items():
    data = hist_dicts_gender[v]
    c = colors[v]
    # print(data)
    axs[v].hist(data, weights=np.ones(len(data)) / data.shape[0], color=c)
    axs[v].set_title(k)
plt.savefig(f'{arg["outdir"]}/fig2_gender.png')

# plot race
fig, axs = plt.subplots(1, len(race_to_idx), sharey=True, tight_layout=True)
print(race_to_idx)
fig.suptitle("Race ")
fig.supxlabel("Label Homogeneity")
fig.supylabel("Fraction of Individuals")
for k,v in race_to_idx.items():
    c = colors[v]
    data = hist_dicts_race[v]
    axs[v].hist(data, weights=np.ones(len(data)) / data.shape[0], color=c)
    axs[v].set_title(k)
plt.savefig(f'{arg["outdir"]}/fig2_race.png')
