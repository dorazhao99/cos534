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
    parser.add_argument('--humanlabels', type=str, default=None)
    parser.add_argument('--modelpath', type=str, default=None)
    parser.add_argument('--labels_test', type=str, default=None)
    parser.add_argument('--batchsize', type=int, default=170)
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--outfile', type=str)
    arg = vars(parser.parse_args())
    print(arg, '\n', flush=True)
    
    humanlabels = json.load(open(arg['humanlabels']))
    labels = pickle.load(open(arg['labels_test'], 'rb'))

    # Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = Classifier(device=device, dtype=arg['dtype'],
                            num_classes=arg['num_classes'], 
                            input_size=256, 
                            modelpath=arg['modelpath'])
    
    # Create dataloader
    testset = create_dataset(labels_path=arg['labels_test'], batch_size=arg['batchsize'], train=False)

    # Do inference with the model
    loss, corrects, y_preds, y_true = classifier.test(testset)
    y_preds = torch.cat(y_preds); y_true = torch.cat(y_true)
    y_preds = y_preds.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    print(y_preds.shape, y_true.shape, flush=True)
    
    # Print out total accuracy
    acc = 100 * corrects.double() / len(testset.dataset)
    print("Total accuracy: {:.2f}%".format(acc), flush=True)

    # Get confusion matrix and normalize diagonal entries
    cm = confusion_matrix(y_true, y_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    for label in humanlabels:
        idx = humanlabels[label]
        print("Accuracy for {}: {:.2f}".format(label, 100.0 * cm.diagonal()[idx]), flush=True)

    # Print results to file
    output = {'pred': list(y_preds.tolist()), 'true': np.stack(y_true).tolist(), 'labels': [x for x in labels]}
    with open(arg['outfile'], 'w') as f:
        json.dump(output, f)
    
if __name__ == "__main__":
    main()
