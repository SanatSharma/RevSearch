import os
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.model_selection import train_test_split


def extract_keypoints(imgs, label):
    keypoints = np.array([sift_extract(i) for i in imgs])
    path = "./sift_keypoints/"
    if not os.path.exists(path):
        os.makedirs(path)
    np.save('./sift_keypoints/{}_keypoints'.format(label), keypoints)
    print("Saved keypoints to ./sift_keypoints/{}_keypoints.npy")

def load_keypoints(label):
    return np.load('./sift_keypoints/{}_keypoints.npy'.format(label))
    
def sift_extract(fig, plot=False):
    fig = np.array(fig, dtype=np.uint8)
    grey = cv2.cvtColor(fig, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, dsc = sift.detectAndCompute(grey,None)
    if plot:
        display = cv2.drawKeypoints(grey, kp, fig)
        plt.imshow(display)
        plt.show()
    return dsc


class SIFT:
    
    def __init__(self):
        self.retrieve_size = 10
        self.train_keypoints, self.test_keypoints = None, None
        self.retrieved_idx, self.retrieved_labels = None, None
        self.query_label, self.compare_labels = None, None
        self.best_ratio = None
        self.report = None
        
    def fit(self, X_train, Y_train, train_keypoints, plot=False):
        self.train_keypoints = train_keypoints
        # find Lowe's ratio with highest F-1 score on train set
        train_scores = []
        ratios = [(i+1)*0.05 for i in range(20)]
        for r in ratios:
            s = self.__iterate_images(images=X_train, labels=Y_train, ratio=r, metric='f1', step='train')
            train_scores.append(np.mean(s))
        self.best_ratio = ratios[np.argmax(train_scores)]
        if plot:
            plt.plot(ratios, train_scores)
            plt.xlabel("Lowe's ratio"); plt.ylabel(metric); plt.grid()
    
    def predict(self, X_test, Y_test, test_keypoints, metric='f1'):
        if not self.best_ratio:
            print("Call fit() first")
            raise ValueError
        self.test_keypoints = test_keypoints
        score = self.__iterate_images(X_test, Y_test, self.best_ratio, metric, step='test')
        print(len(X_test), metric, score)

    def __iterate_images(self, images, labels, ratio, metric, step):
        scores = []
        kps = self.train_keypoints if step=='train' else self.test_keypoints
        # i == current query image index
        for i in range(images.shape[0]):
            query_kps = kps[i]
            compare_kps_collection = np.delete(kps, i)
            self.query_label, self.compare_labels = labels[i], np.delete(labels, i)
            report = self.find_similar_images(query_kps, compare_kps_collection, ratio)
            scores.append(report[metric])
        return scores

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
        
    def plot_scores(self, ratios):
        fig, axarr = plt.subplots(3, sharex=True)
        axarr[0].plot(ratios, precisions)
        axass[0].set_xlabel("Lowe's ratio"); axass[0].set_ylabel("precision")
        axarr[1].plot(ratios, recalls)
        axass[1].set_xlabel("Lowe's ratio"); axass[1].set_ylabel("recall")
        axarr[2].plot(ratios, f1s)
        axass[2].set_xlabel("Lowe's ratio"); axass[0].set_ylabel("F1 score")
        plt.show()