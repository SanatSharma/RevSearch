# define models here

import torch
import torchvision
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from search_data import *
import numpy as np
from sklearn.decomposition import PCA
import pickle


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Neural(nn.Module):
    def __init__(self):
        super(Neural, self).__init__()
        # Use pretrained resnet model
        resnet = models.vgg11(pretrained=True)
        self.new_resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        # Decompose new_resnet features into a vector of size 300
        self.pca = PCA(n_components=300)
        print(self.new_resnet)

    def forward(self, x, batch_size=1):
        feats = self.new_resnet(x)
        return feats

    def reduce_dimensionality(self, feats):
        self.pca.fit(features)
        reduced_feats = self.pca.transform(features)
        return reduced_feats

def train_neural_model(train_data, test_data):
    model = Neural()
    model.to(device)
    model.eval()
    epochs = 1
    batch_size = 10
    neural_net_output_size = 512
    with torch.no_grad():
        for epoch in range(epochs):
            print(str(epoch) + " of " + str(epochs) + " epochs")
            neural_feats = torch.zeros([len(train_data)*batch_size, neural_net_output_size])
            for batch_idx, (inputs, outputs) in enumerate(train_data):
                if (batch_idx%100==0):
                    print(str(batch_idx) + " of " + str(len(train_data)) + " examples")
                inputs = inputs.to(device)
                feats = model.forward(inputs)
                for i in range(len(feats)):
                    neural_feats[batch_idx*batch_size + i] = feats[i,:,0,0]

            print(neural_feats)
            torch.save(neural_feats, 'torch.pt')



    raise Exception("Implement training")
