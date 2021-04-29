import time, random
import numpy as np
import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.optim as optim

class ResNet50(nn.Module):
    def __init__(self, n_classes, input_size, pretrained=True, feature_extract=False, hidden_size=2048):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.require_all_grads(feature_extract)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, n_classes)
        self.input_size = input_size

    def require_all_grads(self, feature_extract):
        for param in self.parameters():
            param.requires_grad = feature_extract

    def forward(self, x):
        outputs = self.resnet(x)
        return outputs
    
class Classifier():
    def __init__(self, device, dtype, num_classes, input_size=256, lr=0.001, modelpath=None):
        self.device = device     
        self.dtype = dtype
        self.num_classes = num_classes
        self.model = ResNet50(n_classes=num_classes, input_size=input_size)
        self.num_ftrs = self.model.resnet.fc.in_features
        self.model.resnet.fc = nn.Linear(self.num_ftrs, self.num_classes)

        feature_extract = False        

        # Observe that all parameters are being optimized
        params_to_update = self.model.parameters()
        if feature_extract:
            params_to_update = []
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)
            self.model = self.model.to(device=self.device, dtype=self.dtype)

        self.optimizer = optim.Adam(params_to_update, lr=lr)
        self.criterion = nn.CrossEntropyLoss()        

        self.print_freq = 50
        self.epoch = 0
        if modelpath != None:
            self.model.load_state_dict(torch.load(modelpath, map_location=self.device)) # Need to check
        
    def train(self, dataloader):
        # Set model to train mode
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.train()
        
        # Iterate over data.
        total_loss = 0.0; corrects = 0
        for i, (inputs, labels, _) in enumerate(dataloader):
            start = time.time()
            inputs = inputs.to(device=self.device, dtype=self.dtype)
            labels = labels.to(device=self.device, dtype=self.dtype)
            labels = labels.type(torch.LongTensor)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward batch
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                # backward + optimize
                loss.backward()
                self.optimizer.step()
                
            # Keep track of loss
            total_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)
            
            if i % self.print_freq == 0:
                print("Iter {} (Epoch {}), Train Loss = {:.3f}".format(i, self.epoch, loss.item()))            
        return total_loss, corrects
            
                
    def test(self, dataloader):
        # Set model to eval mode
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        
        # Iterate over data.
        total_loss = 0.0; corrects = 0
        with torch.no_grad():
            for i, (inputs, labels, _) in enumerate(dataloader):
                start = time.time()
                inputs = inputs.to(device=self.device, dtype=self.dtype)
                labels = labels.to(device=self.device, dtype=self.dtype)
                labels = labels.type(torch.LongTensor)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward batch
                # track history if only in train
                with torch.set_grad_enabled(False):
                    # Get model outputs and calculate loss
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    # print(loss)
                    _, preds = torch.max(outputs, 1)

                # Keep track of total corrects
                total_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)
        return total_loss, corrects, preds
