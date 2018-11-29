# utils.py This file may be used for all utility functions
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import math
from PIL import Image
'''
 Create a bijection betweeen int and object. May be used for reverse indexing
'''
class Indexer(object):
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.ints_to_objs)

    def contains(self, obj):
        return self.index_of(obj) != -1
    
    def index_of(self, obj):
        if obj in self.objs_to_ints:
            return self.objs_to_ints[obj]
        return -1
    
    def get_object(self, idx):
        if idx in self.ints_to_objs:
            return self.ints_to_objs[idx]
        return -1

    # Get the index of the object, if add_object, add object to dict if not present
    def get_index(self, obj, add_object = True):
        if not add_object or obj in self.objs_to_ints:
            return self.index_of(obj)
        new_idx = len(self.ints_to_objs)
        self.objs_to_ints[obj] = new_idx
        self.ints_to_objs[new_idx] = obj
        return new_idx

# Add features from feats to feature indexer
# If add_to_indexer is true, that feature is indexed and added even if it is new
# If add_to_indexer is false, unseen features will be discarded
def add_dataset_features(feats, feature_indexer):
    for i in range(len(feats)):
        feature_indexer.get_index(feats[i][0])

def get_concatentated_images(indexes, image_database):
    result = []
    for idx in indexes:
        result.append(image_database[idx][0])
    return result

def find_cosine_distance(a, b):
    sim = cdist(a.reshape(1,-1), b.reshape(1,-1), 'cosine').reshape(-1)[0]
    return sim

def show_similar_images(query_img, retrieved_images):
    query = plt.figure(figsize=(2,2)).add_subplot(111)
    query.imshow(Image.open(query_img))
    query.set_xlabel("query image")
    retrieved = plt.figure()
    row, col = math.floor(len(retrieved_images)//5), 5
    for i in range(row*col):
        print(retrieved_images[i])
        img = Image.open(retrieved_images[i])
        subplot = retrieved.add_subplot(row,col,i+1)
        plt.axis('off')
        plt.imshow(img)
    plt.show()