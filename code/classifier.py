import time, random
import numpy as np
import torch
import torchvision
from torchvision import models
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self, n_classes, input_size, pretrained=True, feature_extract=False, hidden_size=2048):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.require_all_grads(feature_extract)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.input_size = input_size

    def require_all_grads(self, feature_extract):
        for param in self.parameters():
            param.requires_grad = feature_extract

    def forward(self, x):
        outputs = self.resnet(x)
        return outputs
    
class Classifier():
    def __init__(self, device, dtype, optimizer, criterion, num_classes=1, input_size=256, modelpath=None):
        self.device = device     
        self.dtype = dtype
        self.model = ResNet50(n_classes=num_classes, input_size=input_size)
        model.require_all_grads(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device=self.device, dtype=self.dtype)
        
        self.print_freq = 100
        self.epoch = 0
        if modelpath != None:
            self.model.load_state_dict(torch.load(modelpath, map_location=self.device)) # Need to check
        
    def train(self, dataloader, criterion):
        # Set model to train mode
        self.model.train()
        
        # Iterate over data.
        loss = 0.0
        for i, (inputs, labels, _) in enumerate(dataloader):
            start = time.time()
            inputs = inputs.to(device=self.device, dtype=self.dtype)
            labels = labels.to(device=self.device, dtype=self.dtype)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward batch
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                # print(loss)
                _, preds = torch.max(outputs, 1)

                # backward + optimize
                loss.backward()
                optimizer.step()
                
            # Keep track of loss
            loss += loss.item() * inputs.size(0)
            corrects = torch.sum(preds == labels.data)
            
            if i % self.print_freq == 0:
                print("Iter {} (Epoch {}), Train Loss = {:.3f}".format(i, self.epoch, loss.item()))
            
        return loss, corrects
            
                
    def test(self, dataloader, criterion):
        # Set model to eval mode
        self.model.eval()
        
        # Iterate over data.
        corrects = 0
        for i, (inputs, labels, _) in enumerate(dataloader):
            start = time.time()
            inputs = inputs.to(device=self.device, dtype=self.dtype)
            labels = labels.to(device=self.device, dtype=self.dtype)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward batch
            # track history if only in train
            with torch.set_grad_enabled(False):
                # Get model outputs and calculate loss
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                # print(loss)
                _, preds = torch.max(outputs, 1)
                
            # Keep track of total corrects
            loss += loss.item() * inputs.size(0)
            corrects = torch.sum(preds == labels.data)
        
        return loss, corrects