import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import matplotlib.pyplot as plt

# Creiamo il seguente dataset
np.random.seed(0)  # Per rendere il risultato riproducibile
x = np.linspace(1700, 3700, 100) 
noise = np.random.normal(0, 100, x.size)
y = 0.0001 * (x - 2500)**3 + 100 + noise
x = np.expand_dims(x, axis=1); y = np.expand_dims(y, axis=1)

# A questo punto dividiamo i due dataset in training, cross-validation e test
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)  ## 40% test
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)
del x_, y_  ## Delete temporary variables
scaler_linear = StandardScaler()  ## Classe presente in scikit-learn
X_train_scaled = scaler_linear.fit_transform(x_train)  ## Mean and STDof the training set then transform it

# Usiamo la classe Linear_Regression() presente in scikit-learn
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train )

# A questo punto valutiamo la bontà del modello (MSE = mean standard error)
yhat = linear_model.predict(X_train_scaled)
print(f"training MSE: {mean_squared_error(y_train, yhat) / 2}")


# Fin qui abbiamo usato solo un modello (lineare)
# Aggiungiamo un modello polinomiale
poly = PolynomialFeatures(degree=2, include_bias=False)  
X_train_mapped = poly.fit_transform(x_train)  ## Compute the number of features and transform the training set
scaler_poly = StandardScaler()  ## Classe su scikit-learn
X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)  ## Mean and STD of the training set then transform it

# A questo punto vogliamo vedere la differenza nell'errore tra train e cross-validation
model = LinearRegression()
model.fit(X_train_mapped_scaled, y_train )
yhat = model.predict(X_train_mapped_scaled)
print(f"Training MSE: {mean_squared_error(y_train, yhat) / 2}")
X_cv_mapped = poly.transform(x_cv)  ## Add the polynomial features to the cross validation set
X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)  ## Scale the cross validation set using the mean and STD of the training set
yhat = model.predict(X_cv_mapped_scaled)
print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")


# Per fare la prova con diversi polinomi insieme
# Initialize lists to save the errors, models, and feature transforms
train_mses = []; cv_mses = []; models = []; polys = []; scalers = []

# Loop over 10 times. Each adding one more degree of polynomial higher than the last.
for degree in range(1,11):
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)
    polys.append(poly)
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)
    model = LinearRegression()
    model.fit(X_train_mapped_scaled, y_train )
    models.append(model)
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    train_mses.append(train_mse)
    X_cv_mapped = poly.transform(x_cv)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat) / 2
    cv_mses.append(cv_mse)
    
# Plot the results
degrees=range(1,11)
plt.plot(degrees, train_mses, label='train', color='blue')
plt.plot(degrees, cv_mses, label='CV', color='green')
plt.legend(); plt.show()
















