import pickle, time, glob, argparse, json
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--labels', type=str, default='race.json')
parser.add_argument('--annotations', type=str, default='bfw_annotations.json')
parser.add_argument('--labels_test', type=str, default='bfw_test.pkl')
parser.add_argument('--labels_val', type=str, default='bfw_val.pkl')
parser.add_argument('--labels_train', type=str, default='bfw_train.pkl')
parser.add_argument('--image_path', type=str, default='/Volumes/G-DRIVE mobile USB-C/cos534/data/BFW/')
arg = vars(parser.parse_args())
print('\n', arg, '\n')

path = arg['image_path']

# Load labels
with open(arg['labels']) as f:
    labels = json.load(f)

# Create a list of image file nakes
with open(path + arg['annotations']) as f:
    annotations = json.load(f)

val = [i for i in annotations if annotations[i]['split'] == 'val']
train = [i for i in annotations if annotations[i]['split'] == 'train']
test = [i for i in annotations if annotations[i]['split'] == 'test']

print('train {} val {} test {}'.format(len(train), len(val), len(test)))

train_labels = {}
for file in tqdm(train):
    train_labels[path + file] = np.array(labels[annotations[file]['race']])

print('Finished processing {} train labels'.format(len(train_labels)))
with open(arg['labels_train'], 'wb+') as handle:
    pickle.dump(train_labels, handle)

val_labels = {}
for file in tqdm(val):
    val_labels[path + file] = np.array(labels[annotations[file]['race']])

print('Finished processing {} val labels'.format(len(val_labels)))
with open(arg['labels_val'], 'wb+') as handle:
    pickle.dump(val_labels, handle)

test_labels = {}
for file in tqdm(test):
    test_labels[path + file] = np.array(labels[annotations[file]['race']])

print('Finished processing {} test labels'.format(len(test_labels)))
with open(arg['labels_test'], 'wb+') as handle:
    pickle.dump(test_labels, handle)
