# Come visto nel caso della regressione, possiamo scegliere il numero di neuroni e di
# layer facendo varie prove

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# Aggiungiamo polynomial features
degree = 3
poly = PolynomialFeatures(degree, include_bias=False)
X_train_mapped = poly.fit_transform(x_train)
X_cv_mapped = poly.transform(x_cv)
X_test_mapped = poly.transform(x_test)
scaler = StandardScaler()
X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
X_cv_mapped_scaled = scaler.transform(X_cv_mapped)
X_test_mapped_scaled = scaler.transform(X_test_mapped)


# Per scegliere il modello, creiamo prima 3 modelli in un unica classe
def build_models():
    tf.random.set_seed(20)
    model_1 = Sequential(
        [   Dense(25, activation = 'relu'),
            Dense(15, activation = 'relu'),
            Dense(1, activation = 'linear')],name='model_1')
    model_2 = Sequential(
        [   Dense(20, activation = 'relu'),
            Dense(12, activation = 'relu'),
            Dense(12, activation = 'relu'),
            Dense(20, activation = 'relu'),
            Dense(1, activation = 'linear')],name='model_2')
    model_3 = Sequential(
        [   Dense(32, activation = 'relu'),
            Dense(16, activation = 'relu'),
            Dense(8, activation = 'relu'),
            Dense(1, activation = 'linear')],name='model_3')
    model_list = [model_1, model_2, model_3]   
    return model_list


# Poi:
# Initialize lists that will contain the errors for each model
nn_train_mses = []; nn_cv_mses = []
nn_models = build_models()

# Loop over the the models
for model in nn_models:
    model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),)
    print(f"Training {model.name}...")
    model.fit(
        X_train_mapped_scaled, y_train,
        epochs=300,
        verbose=0)
    print("Done!\n")
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    nn_train_mses.append(train_mse)
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat) / 2
    nn_cv_mses.append(cv_mse)
    
# print results
print("RESULTS:")
for model_num in range(len(nn_train_mses)):
    print(
        f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, " +
        f"CV MSE: {nn_cv_mses[model_num]:.2f}")

# Selezioniamo allora il modello con MSE minore per C-V
model_num = 3
yhat = nn_models[model_num-1].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) / 2

print(f"Selected Model: {model_num}")
print(f"Training MSE: {nn_train_mses[model_num-1]:.2f}")
print(f"Cross Validation MSE: {nn_cv_mses[model_num-1]:.2f}")
print(f"Test MSE: {test_mse:.2f}")








