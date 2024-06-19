import numpy as np
import matplotlib.pyplot as plt


# Function to calculate mean and variance of all features
def estimate_gaussian(X): 
    m, n = X.shape
    mu = 1 / m * np.sum(X, axis = 0)
    var = 1 / m * np.sum((X - mu) ** 2, axis = 0)   
    return mu, var

# Apply the function
mu, var = estimate_gaussian(X_train)   


# Returns the density of the multivariate normal at each data point (row) of X_train
p = multivariate_gaussian(X_train, mu, var)
# Plotting code 
visualize_fit(X_train, mu, var)


# Finds the best threshold to use for selecting outliers based on the results from a validation set (p_val) 
# and the ground truth (y_val)
def select_threshold(y_val, p_val): 
    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predictions = (p_val < epsilon)
        tp = np.sum((predictions == 1) & (y_val == 1)) 
        fp = sum((predictions == 1) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))
        prec = tp / (tp + fp)# Your code here to calculate precision
        rec = tp / (tp + fn)# Your code here to calculate recall
        F1 = 2 * prec * rec / (prec + rec)# Your code here to calculate F1
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_epsilon, best_F1

p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)
print('Best epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set: %f' % F1)

# Find the outliers in the training set 
outliers = p < epsilon
# Visualize the fit
visualize_fit(X_train, mu, var)
# Draw a red circle around those outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize= 10,markerfacecolor='none', markeredgewidth=2)






