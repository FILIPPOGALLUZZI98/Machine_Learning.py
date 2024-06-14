import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Supponiamo di avere il seguente dataset
np.random.seed(42)
x = np.linspace(1700, 3750, 100)
y = 0.25 * X + 100 + np.random.normal(0, 25, 100)
plt.scatter(X, Y, s=10); plt.show()

# Dividiamo i dataset in train e cross validation + test
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)
del x_, y_  ## Per cancellare i dataset iniziali

# Dobbiamo caricare la funzione della regressione lineare che chiamiamo 'linear_model'
linear_model = LinearRegression()
# Train the model
linear_model.fit(X_train_scaled, y_train )

# Per valutare il modello e calcolare mean squared error
yhat = linear_model.predict(X_train_scaled)
mean_squared_error(y_train, yhat)




