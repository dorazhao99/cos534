import pickle, time, argparse, json
from os import path, mkdir
import numpy as np
import torch
import torch.optim as optim
from torchvision import models
from sklearn.metrics import confusion_matrix

from classifier import Classifier
from load_data import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath_gender', type=str, default=None)
    parser.add_argument('--modelpath_race', type=str, default=None)
    parser.add_argument('--humanlabels_gender', type=str, default=None)
    parser.add_argument('--humanlabels_race', type=str, default=None)
    parser.add_argument('--labels_gender', type=str, default=None)
    parser.add_argument('--labels_race', type=str, default=None)
    parser.add_argument('--batchsize', type=int, default=170)
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--num_classes_gender', type=int, default=4)
    parser.add_argument('--num_classes_race', type=int, default=2)
    parser.add_argument('--outfile', type=str)
    arg = vars(parser.parse_args())
    print(arg, '\n', flush=True)
    
    humanlabels_gender = json.load(open(arg['humanlabels_gender']))
    humanlabels_race = json.load(open(arg['humanlabels_race']))
    labels_gender = pickle.load(open(arg['labels_gender'], 'rb'))
    labels_race = pickle.load(open(arg['labels_race'], 'rb'))

    # Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier_gender = Classifier(device=device, dtype=arg['dtype'],
                                   num_classes=arg['num_classes_gender'], 
                                   input_size=256, 
                                   modelpath=arg['modelpath_gender'])
    classifier_race = Classifier(device=device, dtype=arg['dtype'],
                                 num_classes=arg['num_classes_race'], 
                                 input_size=256, 
                                 modelpath=arg['modelpath_race'])
    
    # Create dataloader
    testset_gender = create_dataset(labels_path=arg['labels_gender'], batch_size=arg['batchsize'], train=False)
    testset_race = create_dataset(labels_path=arg['labels_race'], batch_size=arg['batchsize'], train=False)

    # Do inference with the models
    _, _, y_preds_gender, y_true_gender = classifier_gender.test(testset_gender)
    y_preds_gender = torch.cat(y_preds_gender); y_true_gender = torch.cat(y_true_gender)
    y_preds_gender = y_preds_gender.detach().cpu().numpy()
    y_true_gender = y_true_gender.detach().cpu().numpy()
    
    _, _, y_preds_race, y_true_race = classifier_race.test(testset_race)
    y_preds_race = torch.cat(y_preds_race); y_true_race = torch.cat(y_true_race)
    y_preds_race = y_preds_race.detach().cpu().numpy()
    y_true_race = y_true_race.detach().cpu().numpy()

    # Print out total accuracy
    corrects = np.sum((y_preds_gender==y_true_gender) & (y_preds_race==y_true_race))
    acc = 100 * corrects / len(testset_race.dataset)
    print("Total accuracy: {:.2f}%".format(acc), flush=True)

    # Compute confusion matrix
    y_preds = 10. * y_preds_gender + y_preds_race
    y_true = 10. * y_true_gender + y_true_race
    cm = confusion_matrix(y_true, y_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 

    idx = 0
    for g in humanlabels_gender:
        for r in humanlabels_race:
            print('{:<8}, {:<16}: {:.2f}%'.format(g, r, 100. * cm.diagonal()[idx]), flush=True)
            idx += 1

    # Print results to file
    #output = {'pred': list(y_preds.tolist()), 'true': np.stack(y_true).tolist(), 'labels': [x for x in labels]}
    #with open(arg['outfile'], 'w') as f:
    #    json.dump(output, f)
    
if __name__ == "__main__":
    main()
