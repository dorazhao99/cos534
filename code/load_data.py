import pickle
import glob
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from skimage import transform

class Dataset(Dataset):
    def __init__(self, img_keys, img_paths, img_labels, transform=T.ToTensor()):
        self.img_keys = img_keys
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        ID = self.img_paths[index]
        img = Image.open(ID).convert('RGB')
        X = self.transform(img)
        y = self.img_labels[self.img_keys[index]]

        return X, y, ID

def create_dataset(dataset, img_path, labels_path, batch_size=8, train=True):
    img_labels = pickle.load(open(labels_path, 'rb'))
    img_keys = sorted(list(img_labels.keys()))
    img_paths = [img_path + x for x in sorted(list(img_labels.keys()))]
    print(img_paths)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # NO TRANSFORMATIONS ATM
    if train:
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            normalize
        ])        
        shuffle = True
    else:
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            normalize
        ])
        shuffle = False    
    dset = Dataset(img_keys, img_paths, img_labels, transform)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    
    return loader 
