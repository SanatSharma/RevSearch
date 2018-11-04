import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.model_selection import train_test_split

class SIFT:
    
    def __init__(self):
        self.retrieve_size = 10
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None
        self.keypoints = None
        self.retrieved_idx = None
        self.retrieved_labels = None
        self.query_label = None
        self.compare_labels = None
        self.report = None
        self.best_ratio = None

    def find_similar_images(self, query_kps, compare_kps_collection, lowes_ratio):
        # count matches
        match_count = {}
        for i, compare_kps in enumerate(compare_kps_collection):
            count = 0
            if compare_kps is not None:
                self.count_keypoint_match(query_kps, compare_kps, lowes_ratio)
            match_count[i] = count
        # find images with the most matches
        ranked = sorted(match_count.items(), key=lambda kv:kv[1], reverse=True)
        self.retrieved_idx = [ranked[i][0] for i in range(self.retrieve_size)]
        self.retrieved_labels = [self.compare_labels[i] for i in self.retrieved_idx]
        # report scores
        self.evaluate()
        return self.report
    
    def extract_keypoints(self, imgs):
        self.keypoints = np.array([self.sift_extract(i) for i in imgs])
    
    def sift_extract(self, fig, plot=False):
        fig = fig.astype(np.uint8)
        grey = cv2.cvtColor(fig, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, dsc = sift.detectAndCompute(grey,None)
        if plot:
            display = cv2.drawKeypoints(grey, kp, fig)
            plt.imshow(display)
            plt.show()
        return dsc
    
    def count_keypoint_match(self, query_kps, compare_kps, lowes_ratio):
        count = 0
        if np.isnan(compare_kps).all():
            return count
        for kp1 in query_kps:
            if kp1 is None: continue
            pairwise_dist = []
            # Euclidean dist
            for kp2 in compare_kps:
                if kp2 is not None:
                    d = norm(kp1-kp2)
                    pairwise_dist.append(d)
            if all(n == 0 for n in pairwise_dist) or len(pairwise_dist) == 1:
                count += 1
                continue
            # find 2 nearest neighbors
            dist = np.sort(np.asarray(pairwise_dist))
            nn1 = dist[0]
            nn2 = dist[1]
            # ratio test
            if nn1 / float(nn2) < lowes_ratio:
                count += 1
        return count

    def evaluate(self):
        all_relevant = list(self.compare_labels).count(self.query_label)
        retrieved_relevant = self.retrieved_labels.count(self.query_label)
        retrieved_labels = [self.compare_labels[i] for i in self.retrieved_idx]
        precision = retrieved_relevant / self.retrieve_size
        recall = retrieved_relevant / all_relevant
        f1 = 0 if (precision+recall) == 0 else 2*(precision*recall)/(precision+recall)
        self.report = {'class': self.query_label, 
                       'precision': precision, 
                       'recall': recall, 
                       'f1': f1}

    def show_similar_images(self):
        query = plt.figure(figsize=(2,2)).add_subplot(111)
        query.imshow(self.query_img)
        query.set_xlabel("query image")
        retrieved = plt.figure()
        row, col = math.floor(self.retrieve_size//5), 5
        for i in range(row*col):
            img = compare_img[self.retrieved_idx[i]]
            subplot = retrieved.add_subplot(row,col,i+1)
            plt.axis('off')
            plt.imshow(img)
        plt.show()
        
    def fit(self, X_train, Y_train):
        # extract keypoints from training data
        self.X_train, self.Y_train = X_train, Y_train
        self.extract_keypoints(self.X_train)
        # find Lowe's ratio with highest F-1 score
        ratios = [(i+1)*0.05 for i in range(20)]
        f1 = []
        for r in ratios:
            train_f1 = self.__iterate_images(images=self.X_train,
                                             labels=self.Y_train, 
                                             ratio=r, metric='f1')
            f1.append(np.mean(train_f1))
        self.best_ratio = ratios[np.argmax(f1)]
    
    def predict(self, X_test, Y_test, metric='f1'):
        self.X_test, self.Y_test = X_test, Y_test
        return self.__iterate_images(X_test, Y_test, self.best_ratio, metric)
        
    def __iterate_images(self, images, labels, ratio, metric):
        scores = []
        # i: current query image idx
        for i in range(images.shape[0]):
            query_kps = self.keypoints[i]
            compare_kps_collection = np.delete(self.keypoints, i)
            self.query_label, self.compare_labels = labels[i], np.delete(labels, i)
            report = self.find_similar_images(query_kps, compare_kps_collection, ratio)
            scores.append(report[metric])
        return scores

