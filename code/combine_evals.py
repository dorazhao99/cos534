# Combines gender and race labels from two classifier predictions into one json file

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--race_evals', type=str, required=True)
parser.add_argument('--gender_evals', type=str, required=True)
parser.add_argument('--outfile', type=str, default='race_gender_eval.json')
arg = vars(parser.parse_args())

race_preds = json.load(open(arg['race_evals']))
gender_preds = json.load(open(arg['gender_evals']))

race_gender_eval = {'pred': [], 'true': []}
for i,pred in enumerate(gender_preds['pred']):
    race_gender_eval['pred'].append((pred, race_preds['pred'][i]))
for i,pred in enumerate(gender_preds['true']):
    race_gender_eval['true'].append((pred, race_preds['true'][i]))

with open(arg['outfile'], 'w') as f:
    json.dump(race_gender_eval, f)

