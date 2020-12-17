import numpy as np
import os 
import sys 
import imageio
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances

# Random initialization lambda function for cluster centers.
r = lambda: np.random.random(size=1)

class Cluster:
    """ 
    Generic cluster class. Stores the points mean
    previous points and current points.
    """
    def __init__(self, cluster_mu):
        self.cluster_mu = cluster_mu
        self.pts = []
        self.prev_pts = []
        

class ColorKMeans:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.clusters = []
        self.losses = []

        # generate initial clusters randomly, r, g, b vals are randomly initialized 
        # between 0 and 1.
        for _ in np.arange(n_clusters):
            self.clusters.append(Cluster(np.squeeze(np.array([r(), r(), r()]))))
        
    def fit(self, X):
        self.X = X
        self.n_iters = 0
        fit = False 
        while not fit:
            # Assign the points to clusters using euclidean distance.
            self._assign_pts_to_clusters(self.X)
            # If there are any clusters that have points changed between iterations, this means the algorithm has not
            # converged yet. Therefore update cluster centers using r, g, b channels' means.
            if len([c for c in self.clusters if np.array_equal(np.array(c.pts),np.array(c.prev_pts))]) == self.n_clusters:
                print("Iter", self.n_iters, "sqrt loss:", self._calculate_distortion())
                fit = True
                self._update_cluster_centers(reset=False)
            else:
                print("Iter", self.n_iters, "sqrt loss:", self._calculate_distortion())
                self._update_cluster_centers()

            self.n_iters += 1

    def _assign_pts_to_clusters(self, X):
        # Assigns each point to the closest cluster using euclidean distance as the measure.
        self.cluster_centers_ = np.squeeze(np.array([np.array(cc.cluster_mu).T for cc in self.clusters]))
        assignment_labels = pairwise_distances_argmin(X, self.cluster_centers_)

        for label in np.unique(assignment_labels):
            self.clusters[label].pts = X[assignment_labels == label]
        return 

    def _update_cluster_centers(self, reset=True):
        # Updates cluster means using assigned points.
        for cluster in self.clusters:
            cluster.prev_pts = cluster.pts
            pts_arr = np.array(cluster.pts)
            if pts_arr.shape[0] > 0:
                pts_means = pts_arr.mean(axis=0) 
                cluster.cluster_mu = pts_means

            if reset:
                cluster.pts = []
        
    def predict(self, X):
        # Do pointwise prediction by assigning the input point to the
        # closest cluster by looking at the Euclidean distance metric.
        self.cluster_centers_ = np.squeeze(np.array([np.array(cc.cluster_mu).T for cc in self.clusters]))
        labels = pairwise_distances_argmin(X, self.cluster_centers_)
        return labels

    def _calculate_distortion(self):
        # Total euclidean distance over all clusters.
        self._J = 0
        for cc in self.clusters:
            if np.array(cc.pts).shape[0] > 0:
                self._J += np.sum(np.squeeze(pairwise_distances(np.expand_dims(np.array(cc.cluster_mu), axis=1).T, np.array(cc.pts)).T))
        self.losses.append(self._J)
        return self._J

    def get_losses(self):
        return self.losses


def kmeans(image:str) -> None:
    img = imageio.imread(image)
    img = np.array(img, dtype=np.float64) / 255.0
    img = img[:,:,:3]
    flat_img = img.reshape(img.shape[0]*img.shape[1],3)

    # Sklearn sanity check. 
    # kmeans = KMeans(n_clusters=3, random_state=0)
    # kmeans.fit(flat_img)
    # labels = kmeans.predict(flat_img)
    # labeled_img = kmeans.cluster_centers_[labels]
    losses = []
    for n_clusters in [3, 5, 7]:
        kmeans = ColorKMeans(n_clusters=n_clusters)
        kmeans.fit(flat_img)
        labels = kmeans.predict(flat_img)
        losses.append(kmeans.get_losses()) 
        labeled_img = kmeans.cluster_centers_[labels]
        labeled_img = labeled_img.reshape(img.shape[0],img.shape[1],3)
        plt.imshow(labeled_img)
        plt.show()
        plt.clf()
        del kmeans, labeled_img

    plt.plot(np.arange(len(losses[0])), losses[0], 'b-',
            np.arange(len(losses[1])), losses[1], 'r-',
            np.arange(len(losses[2])), losses[2], 'g-')
    plt.title("Per iteration distortion of KMeans")
    plt.xlabel("Iterations")
    plt.ylabel("Distortion")
    plt.legend(['3 clusters', '5 clusters', '7 clusters'])
    plt.show()
    plt.clf()
    return 

if __name__ == "__main__":
    kmeans("data/umn_csci.png")