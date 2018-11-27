from utils import *
import numpy as np
from scipy.cluster.vq import kmeans2
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from collections import Counter
import operator
import math
# Loop over all training images
# Read in sift features for training images
# construct textons using kmeans 
# For each image in the training set, find the closest texton for each sift feature
# Question: How to find a limit of how many words should be in a neighborhood space?
# Save word to image in database

# Test time
# Given a new image, extract words 
# For each word, get the corresponding images. Create an image -> word db
# The image with the maximum amount of words should rank first.
# For images with same amount of words, select randomly?

class MLModel:
    def __init__(self, word_indexer, image_indexer, word_to_image, sift_features):
        self.word_indexer = word_indexer
        self.image_indexer = image_indexer
        self.word_to_image = word_to_image
        self.sift_features = sift_features

    def evaluate (self, test_data):
        weighted_words = get_weighted_words(self.word_to_image)

        visual_words = np.array([self.word_indexer[key] for key in self.word_indexer.keys()])
        print(visual_words.shape)

        for idx, image_idx in enumerate(test_data):
            print(image_idx)
            image_sift_features = self.sift_features[1][image_idx]
            image_word_list = find_closest_words(image_sift_features, visual_words)

            # Construct word frequencies
            word_freqs = Counter(image_word_list)

            for key in word_freqs.keys():
                word_freqs[key] = word_freqs[key] * weighted_words[key]
            freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
            print ("FREQS")
            print(freqs)

        pass

def get_weighted_words(word_to_image):
    weights = []

    # Do tf-idf weighting to weight the words
    N = len(word_to_image.keys())
    for key in word_to_image.keys():
        if len(word_to_image[key]) == 0:
            weights.append(0)
        else:
            tf = 1/N
            idf = math.log10(N / len(word_to_image[key])) 
            weights.append(tf*idf)
    
    # penalize negative weights by reducing half of the value
    for i, val in enumerate(weights):
        if val < 0:
            weights[i] = weights[i]/4

    s = sum([abs(i) for i in weights])
    norm = [abs(i)/s for i in weights]
    return norm

# Train machine learning model
def train_ml_model(train_data, image_indexer, sift_features, args):
    visual_words, _ = kmeans2(sift_features[0], args.num_clusters)

    # Construct index -> word indexer
    word_dict = {}
    for i, word in enumerate(visual_words):
        word_dict[i] = word

    # Word to index dictionary
    word_to_image = {}
    for i in range(args.num_clusters):
        word_to_image[i] = set()


    # Construct inverted file index of word_index to image_index list
    for idx, image_idx in enumerate(train_data):
        print(image_idx)
        image_sift_features = sift_features[1][image_idx]
        print(image_sift_features.shape)
        print(len(image_sift_features))
        image_word_list = find_closest_words(image_sift_features, visual_words)
        c = set()
        for i in image_word_list: c.add(i)
        print("NUM UNIQUE ELEMENTS:", len(c))
        
        # Construct word frequencies
        word_freqs = Counter(image_word_list)
        freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
        #print(freqs)
        for i in freqs[:10]:
            if i[1] >= .05*len(image_sift_features):
                a = word_to_image[i[0]]
                a.add(image_idx)
                word_to_image[i[0]] = a
    
    print(word_to_image)
    return MLModel(word_dict,image_indexer,word_to_image,sift_features)

# Find closest word for each sift_feature. 
def find_closest_words(image_sift_features, mean_features):
    result = []
    for target in image_sift_features:
        # Find similarity coefficient
        #sim = [target.dot(feat)/(norm(target) * norm(feat)) for feat in mean_features]
        #sim = np.array(sim)

        # Distance based 
        sim = [cdist(feat.reshape(1,-1), target.reshape(1,-1), 'cosine').reshape(-1)[0] for feat in mean_features]
        #print(max(sim))
        sim = np.array(sim)
        result.append(sim.argmax())
    return result