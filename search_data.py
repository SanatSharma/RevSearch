# search_data.py
# Construct and preprocess data for model

import torch
import torchvision
from torchvision import transforms, datasets
from utils import *

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
