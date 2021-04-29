import pickle, time, argparse
from os import path, mkdir
import numpy as np
import torch
from torchvision import models

from classifier import Classifier
from load_data import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_gender', action='store_true', default=False)
    parser.add_argument('--modelpath', type=str, default=None)
    parser.add_argument('--labels_test', type=str, default=None)
    parser.add_argument('--batchsize', type=int, default=170)
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--dtype', default=torch.float32)
    arg = vars(parser.parse_args())
    print(arg, '\n', flush=True)

    # Initialize the model
    optimizer_ft = optim.Adam(params_to_update, lr=arg['lr'])
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = Classifier(device=device, dtype=arg['dtype'],
                            num_classes=arg['num_classes'], 
                            input_size=256, 
                            modelpath=arg['modelpath'])
    
    # Create dataloader
    testset = create_dataset(arg['labels_test'], B=arg['batchsize'], train=False)

    # Do inference with the model
    loss, corrects = classifier.test(testset, criterion)
    acc = corrects.double() / len(testset.dataset)
    print("Total accuracy: {4.2f}%".format(acc))

if __name__ == "__main__":
    main()
