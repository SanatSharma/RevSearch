# search_data.py
# Construct and preprocess data for model

import torch
import torchvision
from torchvision import transforms, datasets
from utils import *
import os
import random

def get_cifar_data():
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    
    #index_dataset(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    return trainloader, testloader, trainset

def index_dataset(trainset):
    feature_indexer = Indexer()
    add_dataset_features(trainset, feature_indexer)
    print(len(feature_indexer))

def get_ml_data(train_path):
    indexer = Indexer()
    files = [os.path.join(train_path, p) for p in sorted(os.listdir(train_path))]
    for file in files:
        indexer.get_index(file)

    # Generate training and test set - 95% traning, 5% test
    a = [i for i in range(len(files))]
    random.shuffle(a)
    cutoff = int(len(files)*.05)
    train_data = a[:cutoff]
    test_data = a[cutoff:]
    return train_data, test_data, indexer