import numpy as np
import matplotlib.pyplot as plt

# This function takes the data matrix X and the locations of all centroids inside centroids
# It outputs a one-dimensional array idx that holds the index of the closest centroid to every training example
def closest_centroids(X, centroids):
    K = centroids.shape[0]  ## Set K
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distance = [] 
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i] - centroids[j]) 
            distance.append(norm_ij)
        idx[i] = np.argmin(distance)    
    return idx

# Per applicare la funzione
initial_centroids = np.array([[3,3], [6,2], [8,5]])  ## Selezioniamo a caso
idx = find_closest_centroids(X, initial_centroids)
print("First three elements in idx are:", idx[:3])




# Per cercare i centroidi usando la media
# It recomputes, for each centroid, the mean of the points that were assigned to it
def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):   
        points = X[idx == k]    
        centroids[k] = np.mean(points, axis = 0)  
    return centroids

# Per applicarla
K = 3
centroids = compute_centroids(X, idx, K)
print("The centroids are:", centroids)
compute_centroids_test(compute_centroids)




# Runs the K-Means algorithm on data matrix X, where each row of X is a single 
def kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))
    for i in range(max_iters):
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        idx = find_closest_centroids(X, centroids)
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx

# Set initial centroids
initial_centroids = np.array([[3,3],[6,2],[8,5]])
# Number of iterations
max_iters = 10
# Run K-Means
centroids, idx = kMeans(X, initial_centroids, max_iters, plot_progress=True)




# This function initializes K centroids that are to be used in K-Means on the dataset X
def kMeans_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])    
    centroids = X[randidx[:K]]
    return centroids

# Set number of centroids and max number of iterations
K = 3; max_iters = 10
# Set initial centroids by picking random examples from the dataset
initial_centroids = kMeans_init_centroids(X, K)
# Run K-Means
centroids, idx = kMeans(X, initial_centroids, max_iters, plot_progress=True)



