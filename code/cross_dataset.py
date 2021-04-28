import torch
import time
import tqdm
import numpy as np

from load_data import create_dataset
from utils import update_progress

# input 
#     models - list of strings of model save files
#     dataloaders - list of (dataset, image_path, labels, batch)

def accuracy_generalization_matrix(model_names, datasets):
    num_models = len(model_names)
    num_datasets = len(datasets)
    result = np.zeros((num_models, num_datasets))

    dataloaders = [create_dataset(dataset[0], dataset[1], dataset[2], dataset[3], train=False) for dataset in datasets]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, model_name in enumerate(model_names):
        model = torch.load(model_name)
        model.eval()
        for j, (dataset, dataloader) in enumerate(zip(datasets, dataloaders)):
            start = time.time()

            print(f'Evaluating {model_name} on data {dataset[0]}...')
            for k, (inputs, labels, _) in enumerate(dataloader):
                corrects = 0
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                corrects += torch.sum(preds == labels)
                
                if k % 100 == 0:
                    update_progress(k / len(dataloader.dataset))
            
            accuracy = corrects / len(dataloader.dataset)
            result[i][j] = accuracy

            print(f'Time Elapsed: {time.time() - start:.0f}s, Accuracy: {accuracy:.2f}%')

    return result


def main():
    model_names = []
    datasets = []

    accuracy_generalization_matrix(model_names, datasets)

if __name__ == '__main__':
    main()