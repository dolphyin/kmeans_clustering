import numpy as np
import scipy as sp
import sklearn
import scipy.io
import sklearn
import matplotlib.pyplot as plt

#data = scipy.io.loadmat('data/train_small.mat')
#dataset = data['train'][0][0][0][0]
#data_features = np.transpose(dataset[0].reshape(28*28, dataset[0].shape[2]))

data = scipy.io.loadmat('data/train.mat')
dataset = data['train'][0][0][0]
data_features = np.transpose(dataset.reshape(28*28, dataset.shape[2]))

# TODO: is converging, but sometimes ks alternate
def kmeans(k, input_data):
    #centroids = true_random_centroids(k, input_data)
    centroids = random_centroids(k, input_data)
    cluster_indices = get_cluster_indices(centroids, input_data)
    prev_indices = {i:cluster_indices[i] - cluster_indices[i] for i in xrange(len(cluster_indices))}
    i = 0
    while not is_converged(prev_indices, cluster_indices):
        print(i)
        i+=1
        centroids = get_avg_centroid(centroids, cluster_indices, input_data)
        prev_indices = cluster_indices 
        cluster_indices = get_cluster_indices(centroids, input_data)
    reshaped_centroids = get_reshaped_centroids(centroids, 28,28)
    clusters = [input_data[indices] for indices in cluster_indices.values()]
    return (reshaped_centroids, clusters);

# Returns k random aentroids for the given data
# @params k number of centroinds
#         input_data data to create centroids for
# @return kxd matrix
def random_centroids(k, input_data):
    num_samples = input_data.shape[0]
    indices = np.random.randint(0, num_samples, size=k)
    return { i:input_data[i] for i in range(len(indices))}
    #return input_data[indices]

def true_random_centroids(k, input_data):
    num_samples = input_data.shape[0]
    num_features = input_data.shape[1]
    indices = np.random.randint(255, size=k)
    return {i:np.random.randint(255, size=num_features) for i in range(k)}

def get_random_centroid(input_data): 
    num_features = input_data.shape[1]
    return np.random.randint(255, size=num_features)

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
    closest_centroids = np.argmin(distances, axis=1)
    return {i: np.where(closest_centroids==i)[0] for i in range(k)}

# computes average centroid for each cluster in clusters
# @params clusters k-length list of arrays
# @return array[] centroids 
def get_avg_centroid(centroids, cluster_indices, input_data):
    return { i:get_random_centroid(input_data) if len(cluster_indices[i]) == 0 else \
            np.mean(input_data[cluster_indices[i]],axis=0) for i in cluster_indices.keys()}

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
def is_converged(cluster_indices1, cluster_indices2):
    if len(cluster_indices1) != len(cluster_indices2):
        return False
    for i in range(len(cluster_indices1)):
        isSame = is_indices_equal(cluster_indices1[i], cluster_indices2[i])
        if not isSame:
            return False
    return True

def is_indices_equal(indices1, indices2):
    if indices1.shape != indices2.shape:
        return False
    indices1.sort()
    indices2.sort()
    diff = indices1 - indices2
    if diff.any():
        return False
    return True

def get_reshaped_centroids(centroids, x,y):
    return {label: centroids[label].reshape(x,y) for label in centroids.keys()}

#(centroids, clusters) = kmeans(5, data_features)
