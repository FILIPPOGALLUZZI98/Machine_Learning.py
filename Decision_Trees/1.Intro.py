import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Supponiamo di avere i seguenti dati
X_train = np.array([[1, 1, 1],
[0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0],
 [0, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0]])
y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])


# On each node, we compute the information gain for each feature, then split the node on the feature with 
# the higher information gain, by comparing the entropy of the node with the weighted entropy in the two splitted nodes
def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1- p)*np.log2(1 - p)

# Questa funzione restituisce due liste per i due nodi di split, quello di sinistra l'insieme con 
# feature = 1 e quello di destra con feature = 0
def split_indices(X, index_feature):
    left_indices = []
    right_indices = []
    for i,x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

# Quindi, per esempio
split_indices(X_train, 0)

# Per calcolare la weighted entropy, questa funzione prende lo splitted dataset, i due indici che abbiamo
# splittato e restituisce la weighted entropy
def weighted_entropy(X,y,left_indices,right_indices):
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)    
    weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy

# Per vedere come funziona
left_indices, right_indices = split_indices(X_train, 0)
weighted_entropy(X_train, y_train, left_indices, right_indices)

# A questo punto rimane l'information gain
def information_gain(X, y, left_indices, right_indices):
    p_node = sum(y)/len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X,y,left_indices,right_indices)
    return h_node - w_entropy

# Applicandolo ai dati train
information_gain(X_train, y_train, left_indices, right_indices)

# Per capire quale feature è migliore
for i, feature_name in enumerate(['Feature_1', 'Feature_2', 'Feature_3']):
    left_indices, right_indices = split_indices(X_train, i)
    i_gain = information_gain(X_train, y_train, left_indices, right_indices)
    print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")











