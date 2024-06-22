import numpy as np
import matplotlib.pyplot as plt
import copy, math

# Implementation of z-score
def zscore(X):
    # find the mean of each column/feature
    mu = np.mean(X, axis=0)                
    # find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)                 
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      
    return (X_norm, mu, sigma)

# Normalizziamo i dati
X_norm, X_mu, X_sigma = zscore(X_train)




