import pickle, time, argparse, random
from os import path, makedirs
import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim 
import copy

from load_data import *

def train_model(model, device, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            print(dataloaders['train'])
            for inputs, labels, _ in dataloaders[phase]:
                #print(x)
                #break
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def init_model(num_classes, feature_extract):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = models.resnet50(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 256
        
    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--labels_train', type=str)
    parser.add_argument('--labels_val', type=str)
    parser.add_argument('--labels_test', type=str)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--lr', type=float, default = 0.001)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--device', default=torch.device('cpu'))
    arg = vars(parser.parse_args())
   
    # Initialize the model
    feature_extract = False
    model_ft, input_size = init_model(arg['num_classes'], feature_extract)
   
    # Load Dataset
    trainset = create_dataset(arg['dataset'], arg['image_path'], arg['labels_train'])
    valset = create_dataset(arg['dataset'], arg['image_path'], arg['labels_val'], train=False)
    
    dataloaders_dict = {
        'train': trainset, 
        'val': valset
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=arg['lr'])
    criterion = nn.CrossEntropyLoss()
    print(arg['num_epochs'])
    # Train and evaluate
    model_ft, hist = train_model(model_ft, arg['device'], dataloaders_dict, criterion, optimizer_ft, num_epochs=arg['num_epochs'])
    
if __name__ == "__main__":
    main()
