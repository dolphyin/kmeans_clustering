import numpy as np
import scipy as sp
import sklearn
import scipy.io
import sklearn
import matplotlib.pyplot as plt

data = scipy.io.loadmat('data/train_small.mat')
dataset = data['train'][0][1][0][0]
data_features = np.transpose(dataset[0].reshape(28*28, dataset[0].shape[2]))

def kmeans(k, input_data):
    centroids = random_centroids(k, input_data)
    clusters_indices = get_cluster_indices(centroids, input_data)
    clusters = [(i, input_data[clusters_indices[i][0]]) \
            for i in xrange(len(clusters_indices))]
    prev_clusters = [(i, np.zeros([clusters[i][1].shape[0], clusters[i][1].shape[1]])) \
            for i in xrange(len(clusters))] 
    i = 0
    while not isConverged(prev_clusters, clusters):
        print(i)
        i+=1
        centroids = get_avg_centroid(centroids, clusters)
        prev_clusters = clusters
        clusters_indices = get_cluster_indices(centroids, input_data)
        clusters = [(j, input_data[clusters_indices[j]]) for j in xrange(len(clusters_indices))]
    reshaped_centroids = get_reshaped_centroids(centroids, 28,28)
    return (reshaped_centroids, clusters);

# Returns k random centroids for the given data
# @params k number of centroinds
#         input_data data to create centroids for
# @return kxd matrix
def random_centroids(k, input_data):
    num_samples = input_data.shape[0]
    indices = np.random.randint(0, num_samples, size=k)
    return input_data[indices]

# assigns each point to a centroid
# @params centroids kxd array
#         input_data nxd array
# @return array of arrays of indices for each centroid
def get_cluster_indices(centroids, input_data):
    k = len(centroids)
    num_samples = input_data.shape[0]
    distances = np.zeros([num_samples, k])
    for i in range(k):
        distances[:, i] = get_distances(centroids[i], input_data)
    closest_centroids = np.argmax(distances, axis=1)
    return [np.where(closest_centroids==i) for i in range(k)]

# computes average centroid for each cluster in clusters
# @params clusters k-length list of arrays
# @return array[] centroids 
def get_avg_centroid(centroids, clusters):
    return [ centroids[i] if len(clusters[i][1]) == 0 else \
            np.average(clusters[i][1],axis=0) for i in range(len(clusters))]

# gets distances from point to a cluster of points
# @params: point 1xd
#          cluster nxd
# @return  nx1 distances
def get_distances(point, cluster):
    return np.sum((point - cluster)**2, axis=1)

# checks if two clusters are equal
# @params cluster1 array[]
#         cluster2 array[]
# @return bool
def isConverged(clusters1, clusters2):
    if len(clusters1) != len(clusters2):
        return False
    for i in range(len(clusters1)):
        print(clusters1[i][1])
        isSame = isClustersEqual(clusters1[i][1], clusters2[i][1])
        if not isSame:
            return False
    return True

def isClustersEqual(cluster1, cluster2):
    if cluster1.shape != cluster2.shape:
        return False
    print(cluster1.shape)
    sort_cluster1 = cluster1[np.argsort(cluster1)]
    sort_cluster2 = cluster2[np.argsort(cluster2)]
    diff = sort_cluster1 - sort_cluster2
    if diff.any():
        return False
    return True

def get_reshaped_centroids(centroids, x,y):
    return [centroid.reshape(x,y) for centroid in centroids]

