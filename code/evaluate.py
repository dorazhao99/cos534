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
    parser.add_argument('--do_gender', type=int, default=0)
    parser.add_argument('--modelpath', type=str, default=None)
    parser.add_argument('--labels_test', type=str, default=None)
    parser.add_argument('--batchsize', type=int, default=170)
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--num_classes', type=int, default=4)
    arg = vars(parser.parse_args())
    print(arg, '\n', flush=True)

    
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
    loss, corrects, y_preds = classifier.test(testset, criterion)

    # Print out total accuracy
    acc = corrects.double() / len(testset.dataset)
    print("Total accuracy: {4.2f}%".format(acc))

    # Print out per-class accuracy
    labels_list = torch.stack(list(labels.values()))
    print(labels_list.shape, flush=True)
    y_true = list(labels.values())

    # Get confusion matrix and normalize diagonal entries
    cm = confusion_martix(y_true, y_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print(cm)
    #for label in humanlabels_to_onehot:
    #    idx = humanlabels_to_idx[label]
    #    print("Accuracy for {}: {:.2f}".format(label, 100.0 * cm.diagonal()[idx]))

if __name__ == "__main__":
    main()
