import torch
import time
import tqdm
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

from load_data import create_dataset
from utils import update_progress

# input 
#     models - list of strings of model save files
#     dataloaders - list of (dataset, image_path, labels, batch)

def accuracy_generalization_matrix(model_names, datasets, device):
    num_models = len(model_names)
    num_datasets = len(datasets)
    result = np.zeros((num_models, num_datasets))

    dataloaders = [create_dataset(dataset[1], dataset[2], dataset[3], train=False) for dataset in datasets]

    for i, model_name in enumerate(model_names):
        model = torch.load(model_name)
        model = model.to(device)

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

    print(f'---- Cross-Dataset Generalization ----')
    print(result)

    return result


def compute_fleiss_kappa(num_categories, model_names, device, dataloader):
    num_subjects = len(dataloader.dataset)
    fleiss_input = np.zeros((num_subjects, num_categories))

    for model_name in model_names:
        model = torch.load(model_name)
        model = model.to(device)
        model.eval()
        start = time.time()
        
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                for j, output in enumerate(outputs):
                    fleiss_input[i * len(outputs) + j][output] += 1
                
                if i % 100 == 0:
                    update_progress(i / len(dataloader.dataset))

        print(f'Time Elapsed: {time.time() - start:.0f}s, Accuracy: {accuracy:.2f}%')

    return fleiss_kappa(fleiss_input)


def fleiss_kappa_gender(model_names, gender, dataset, device, num_genders=2):
    # WARNING: requires dataloader shuffle to be FALSE
    dataloader = create_dataset(dataset[1], dataset[2], dataset[3], train=False, gender=gender)
    return compute_fleiss_kappa(num_genders, model_names, device, dataloader)
    


def fleiss_kappa_race(model_names, race, dataset, device, num_races=4):
    # WARNING: requires dataloader shuffle to be FALSE
    dataloader = create_dataset(dataset[1], dataset[2], dataset[3], train=False, race=race)
    return compute_fleiss_kappa(num_races, model_names, device, dataloader)


def fleiss_kappa_gender_matrix(model_names, datasets, device):
    genders = ['M', 'F']
    result = np.zeros((2, len(datasets)))

    for i, gender in enumerate(genders):
        for j, dataset in enumerate(datasets):
            result[i][j] = fleiss_kappa_gender(model_names, gender, dataset, device)

    print('----- FLEISS KAPPA MATRIX -----')
    print(result)

    return result


def main():
    model_names = []
    datasets = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    accuracy_generalization_matrix(model_names, datasets, device)

    fleiss_kappa_gender_matrix(model_names, datasets, device)

if __name__ == '__main__':
    main()
