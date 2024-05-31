import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Scikit-learn has a gradient descent regression model sklearn.linear_model.SGDRegressor
# Like your previous implementation of gradient descent, this model performs best with normalized inputs
# sklearn.preprocessing.StandardScaler will perform z-score normalization as in a previous lab. Here it is referred to as 'standard score'


# Dataset (che contiene: area, # camere, # piani, età della casa)
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])  ## Prezzo
X_features = ['size(sqft)','bedrooms','floors','age']

# Scale/normalize the training data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

# Create and fit the regression model 
sgdr = SGDRegressor(max_iter=1000) 
sgdr.fit(X_norm, y_train) 

# Make predictions
# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm

# Plot Results
# plot predictions and targets vs original features    
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()























