'''search.py
 This file may be used for kicking off the model, reading in data and preprocessing
 '''

import argparse
import sys
from model import *
from utils import *
from search_data import *

# Read in command line arguments to the system
def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='Neural', help="Model to run")
    parser.add_argument('--train_type', type=str, default="CIFAR10", help="Data type - Cifar10 or custom")
    parser.add_argument('--train_path', type=str, default='../data/Reference/', help='Path to the training set')
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

    else:
        raise Exception("Please select appropriate model")
    