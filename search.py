'''search.py
 This file may be used for kicking off the model, reading in data and preprocessing
 '''

import argparse
import sys
from model.convnet import *
from model.inverted_file import *
from model.sift import *
from utils import *
from search_data import *
import os


# Read in command line arguments to the system
def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='ML', help="Model to run")
    parser.add_argument('--train_type', type=str, default="CIFAR10", help="Data type - Cifar10 or custom")
    parser.add_argument('--train_path', type=str, default='data/Reference/', help='Path to the training set')
    parser.add_argument('--sift_path', type=str, default='model/sift.npy', help='Path to the training set')
    parser.add_argument('--num_clusters', type=int, default=100, help='Number of kmeans clusters for traditional ML model')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    print(args)

    if (args.model == 'Neural'):
        # Get images from train path and call CNN model train function
        if (args.train_type == 'CIFAR10'):
            train_data, test_data, image_database = get_cifar_data()
            print("training")
            model, neural_feats =  train_neural_model(train_data)
            print("testing")            
            evaluate(model, test_data, neural_feats, image_database)

    elif args.model == 'ML':
        if args.train_type == 'CUSTOM':
            if not os.path.isfile(args.sift_path):
                generate_sift_features(args.train_path, args.sift_path)
            sift_features = load_keypoints_all(args.sift_path)
            train_data, test_data, image_indexer = get_ml_data(args.train_path)
            print('training')
            trained_model = train_ml_model(train_data, image_indexer, sift_features, args)
            print('testing')
            trained_model.evaluate(test_data)

    else:
        raise Exception("Please select appropriate model")
    