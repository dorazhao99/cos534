import pickle, time, argparse, random
from os import path, makedirs
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import copy

from classifier import Classifier
from load_data import *
    
def train_model(classifier, outdir, device, dataloaders, num_epochs):
    since = time.time()

    val_acc_history = []; loss_epoch_list = []

    best_model_wts = copy.deepcopy(classifier.model.state_dict())
    best_acc = 0.0

    running_loss = 0.0; running_corrects = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:            
            if phase == 'train':    
                running_loss, running_corrects = classifier.train(dataloaders['train'])
            else:
                running_loss, running_corrects = classifier.test(dataloaders['val'])            
            #epoch_loss = running_loss / len(dataloaders[phase].dataset)
            #epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            epoch_loss = running_loss / (32 * 4)
            epoch_acc = running_corrects.item() / (32 * 4)
            print('{} Loss: {:.4f} Acc: {:.4f}  Time: {:.4f}'.format(phase, epoch_loss, epoch_acc, time.time() - since))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(classifier.model.state_dict())

                # save model
                torch.save(best_model_wts, '{}/model_best.pth'.format(outdir))

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                loss_epoch_list.append(epoch_loss)

                # Save the model
                if (epoch + 1) % 5 == 0:
                    torch.save(classifier.model.state_dict(), '{}/model_{}.pth'.format(outdir, epoch))
                    
        classifier.epoch += 1

    print('Best model at {} with lowest val loss {}'.format(np.argmin(loss_epoch_list), np.min(loss_epoch_list)))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))

    # load best model weights
    classifier.model.load_state_dict(best_model_wts)
    return classifier, val_acc_history

"""
def init_model(num_classes, feature_extract):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
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
"""

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--labels_train', type=str)
    parser.add_argument('--labels_val', type=str)
    parser.add_argument('--labels_test', type=str)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--lr', type=float, default = 0.001)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--outdir', type=str)
    arg = vars(parser.parse_args())

    # Initialize the model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(arg['num_classes'])
    classifier = Classifier(device=device, dtype=arg['dtype'],
                            num_classes=arg['num_classes'], 
                            input_size=256, lr = arg['lr'],
                            modelpath=arg['model_path'])

    # Load Dataset
    trainset = create_dataset(arg['dataset'], arg['image_path'], 
                              arg['labels_train'], arg['batchsize'])
    valset = create_dataset(arg['dataset'], arg['image_path'], 
                            arg['labels_val'], arg['batchsize'], train=False)

    dataloaders_dict = {
        'train': trainset,
        'val': valset
    }

    # Observe that all parameters are being optimized
    #params_to_update = classifier.model.parameters()
    #feature_extract = False
    #if feature_extract:
    #    params_to_update = []
    #    for name,param in classifier.model.named_parameters():
    #        if param.requires_grad == True:
    #            params_to_update.append(param)
    #            print("\t", name)
    #else:
    #    for name,param in classifier.model.named_parameters():
    #        if param.requires_grad == True:
    #            print("\t", name)
    
    # Train and evaluate
    best_classifier, hist = train_model(classifier, arg['outdir'], device,
                                        dataloaders_dict, 
                                        num_epochs=arg['num_epochs'])

if __name__ == "__main__":
    main()
