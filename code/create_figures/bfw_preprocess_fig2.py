import time
import os
import json
import numpy as np
import argparse
import collections
import matplotlib.pyplot as plt
from statsmodels.stats.inter_rater import fleiss_kappa
import csv

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
    category_counts /= len(eval_pred)
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

file_paths = ["/Users/sahanpaliskara/Documents/cos534/results/bfw/bfw_on_bfw_eval.json", "/Users/sahanpaliskara/Documents/cos534/results/bfw/fairface_on_bfw_eval.json"]
ann_path = "/Users/sahanpaliskara/Documents/bfw-v0.1.5-datatable.csv"
gender_path = "/Users/sahanpaliskara/Documents/cos534/data/gender.json"
race_path = "/Users/sahanpaliskara/Documents/cos534/data/race.json"

gender_to_idx = json.load(open(gender_path))
race_to_idx = json.load(open(race_path))
# Get list of all filenames
evals_pred = []; evals_true = []; labels = []
for f in file_paths:
    eval = json.load(open(f))
    evals_pred.append(eval['pred'])
    evals_true.append(eval['true'])
    labels.append(eval['labels'])
file_name_to_id, num_individuals = make_json(ann_path)
hist_dicts_gender = create_histogram(evals_pred, evals_true, labels, file_name_to_id, num_individuals, len(gender_to_idx), category_name='gender')
print(len(hist_dicts_gender[0]) + len(hist_dicts_gender[1]))
print(hist_dicts_gender)


