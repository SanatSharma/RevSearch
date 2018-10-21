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
import matplotlib.pyplot as plt

# Run on gpu is present
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
        print(x.shape)
        feats = self.new_resnet(x)
        return feats

    def reduce_dimensionality(self, feats):
        self.pca.fit(features)
        reduced_feats = self.pca.transform(features)
        return reduced_feats

def train_neural_model(train_data):
    model = Neural()
    model.to(device)
    model.eval()
    epochs = 1
    batch_size = 10
    neural_net_output_size = 512
    try:
        neural_feats = torch.load('torch.pt')
        return model, neural_feats
    except:
        with torch.no_grad():
            for epoch in range(epochs):            
                print(str(epoch) + " of " + str(epochs) + " epochs")   
                neural_feats = torch.zeros([len(train_data)*batch_size, neural_net_output_size])
                for batch_idx, (inputs, outputs) in enumerate(train_data):
                    if (batch_idx%100==0):
                        print(str(batch_idx) + " of " + str(len(train_data)) + " examples")
                    inputs = inputs.to(device)

                    # Make a grid from batch and display images
                    # out = torchvision.utils.make_grid(inputs)
                    #imshow(out)

                    feats = model.forward(inputs, batch_size=batch_size)
                    for i in range(len(feats)):
                        neural_feats[batch_idx*batch_size + i] = feats[i,:,0,0]
                print(neural_feats)
                torch.save(neural_feats, 'torch.pt')
            return model, neural_feats
        
def evaluate(model, test_data, neural_feats, image_database):
    batch_size = 2
    for batch_idx, (inputs, outputs) in enumerate(test_data):
        if batch_idx < 2:
            out = torchvision.utils.make_grid(inputs)
            imshow(out, "Test Images")
            inputs = inputs.to(device)
            feats = model.forward(inputs, batch_size=batch_size)
            for i in range(batch_size):
                indexes = find_closest_images(feats[i,:,0,0], neural_feats)
                print(indexes)
                result_inputs = get_concatentated_images(indexes, image_database)
                
                out = torchvision.utils.make_grid(result_inputs)
                imshow(out, "Results")


def find_closest_images(target, features, n=5):
    print(target.shape)
    print(features.shape)
    # cosine sim(u,v) = dot(u,v)/ (norm(u) * norm(v))
    distances = [target.dot(feat)/(target.norm() * feat.norm()) for feat in features]
    closest_sorted = sorted(range(len(distances)), key=lambda i: distances[i])
    # return the top n closest 
    return closest_sorted[:n+1]

def get_concatentated_images(indexes, image_database):
    result = []
    for idx in indexes:
        result.append(image_database[idx][0])
    return result

# This function was taken from StackOverflow
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()

