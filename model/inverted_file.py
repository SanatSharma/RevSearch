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

def train_ml_model(train_data, image_indexer):
    for batch_idx, (input, output) in enumerate(train_data):
        print("INPUT")
        print(input)
        print("OUTPUT")
        print(output)
