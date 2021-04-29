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
    def __init__(self, img_keys, img_labels, transform=T.ToTensor()):
        self.img_keys = img_keys
        # self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_keys)

    def __getitem__(self, index):
        ID = self.img_keys[index]
        img = Image.open(ID).convert('RGB')
        X = self.transform(img)
        y = self.img_labels[self.img_keys[index]]
        return X, y, ID


def get_image_paths(img_path, img_keys, img_labels, race=None, gender=None):
    if gender is not None and race is not None:
        return [img_path + x for x in img_keys if img_labels[x]['gender'] == gender and img_labels[x]['race'] == race]
    elif gender is not None:
        return [img_path + x for x in img_keys if img_labels[x]['gender'] == gender]
    elif race is not None:
        return [img_path + x for x in img_keys if img_labels[x]['race'] == race]
    else:
        return [img_path + x for x in img_keys]


def create_dataset(labels_path, batch_size, train=True, gender=None, race=None):
    img_labels = pickle.load(open(labels_path, 'rb'))
    img_keys = sorted(list(img_labels.keys()))
    # img_paths = get_image_paths(img_path, img_keys, img_labels, race, gender)
    
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
    dset = Dataset(img_keys, img_labels, transform)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    return loader
