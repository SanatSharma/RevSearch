# RevSearch
Optimized Reverse Image Search for all! This project aims to create a scalable and robust system for reverse Image search. There are 2 types of models implemented: Neural Models, Machine Learning Model.

Initial dataset: CIFAR 10

To run the following packages are needed:
```
torch
torchvision
numpy
sklearn
python3x
```
Running script

```
cd models
python3 search.py
```

This project is highly customizable and you can use it to do reverse image search on your own dataset. Some customizable parameters are:
```
--model, type=str, default='ML', help="Model to run (CONV / ML)"
--train_type, type=str, default="CIFAR10", help="Data type - Cifar10 or CUSTOM"
--train_path, type=str, default='data/Reference/', help='Path to the training set'
--sift_path', type=str, default='model/sift.npy', help='Path to the sift file. Only useful if using ML model'
--num_clusters, type=int, default=100, help='Number of kmeans clusters for traditional ML model'
```

## Models

1) Neural model - The neural model utilizes transfer learning and uses a pre-trained VGG-16 model, trained on the ImageNet dataset for feature-extraction of images. PCA (principal component analysis) is used to reduce the dimensionality of this vector to construct an image dataset of Image id -> feature vector. 

2) ML Model - The traditional machine learning breaks down images into visual words and constructs an inverted file of words -> images for extremely fast retrieval. This model is heavily influenced by (http://www.cs.utexas.edu/~grauman/courses/fall2009/papers/bag_of_visual_words.pdf).
