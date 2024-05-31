import numpy as np
import matplotlib.pyplot as plt

# Dataset (che contiene: area, # camere, # piani, età della casa)
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])  ## Prezzo
X_features = ['size(sqft)','bedrooms','floors','age']

# Per fare grafici che mostrano le varie caratteristiche
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()


#############################################################################################
#############################################################################################
####  LEARNING RATE

# Impostiamo alpha = 9.9e-7
_, _, hist = gradient_descent(X_train, y_train, 10, alpha = 9.9e-7)
plot_cost_i_w(X_train, y_train, hist)

# Impostiamo alpha = 9e-7
_,_,hist = gradient_descent(X_train, y_train, 10, alpha = 9e-7)
plot_cost_i_w(X_train, y_train, hist)

# Impostiamo alpha = 1e-7
_,_,hist = gradient_descent(X_train, y_train, 10, alpha = 1e-7)
plot_cost_i_w(X_train,y_train,hist)


#############################################################################################
#############################################################################################
#### FEATURE SCALING

def zscore_normalize_features(X):
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                 
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      
    return (X_norm, mu, sigma)

# Normalizziamo i dati
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)

# Facciamo i grafici delle varie caratteristiche
fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],X_train[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
plt.show()
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_norm[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count"); 
fig.suptitle("distribution of features after normalization")
plt.show()

# A questo punto rifacciamo il GD
w_norm, b_norm, hist = gradient_descent(X_norm, y_train, 1000, 1.0e-1, )





