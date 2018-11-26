from utils import *
from scipy.cluster.vq import kmeans2
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
    def __init__(self, word_indexer, image_indexer):
        self.word_indexer = word_indexer
        self.image_indexer = image_indexer
    def evaluate (self, test_data):
        pass

def train_ml_model(train_data, image_indexer, sift_features, args):
    image_words, _ = kmeans2(sift_features[0], args.num_clusters)

    # Construct index -> word indexer
    word_indexer = Indexer()
    for i in image_words:
        word_indexer.get_index(i)

    # Construct inverted file index of word_index to image_index list
    