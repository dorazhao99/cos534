import json, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--outfile', type=str)
arg = vars(parser.parse_args())
print('\n', arg, '\n')

res = json.load(open(arg['file']))
pred, true, files = res['pred'], res['true'], res['labels']

gender_wrong, race_wrong = [], []

for i, x in enumerate(pred):
    if x[0] != true[i][0]: gender_wrong.append(files[i])
    if x[1] != true[i][1]: race_wrong.append(files[i])

output = {'gender': gender_wrong, 'race': race_wrong}
json.dump(output, open(arg['outfile'], 'w'))
print(gender_wrong)
print(race_wrong)


