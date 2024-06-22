import numpy as np
import matplotlib.pyplot as plt
import math, copy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Supponiamo di dover prevedere prevedere il consumo di carburante (in litri per 100 km)
# di un'auto in base alla sua velocità media (in km/h).
# Si tratta di una regressione lineare con singola variabile
# Il modello sarà del tipo f_wb = w*x + b
# Per una singola variabile non c'è il la regularization perché l'overfitting non è un problema


#############################################################################################
#############################################################################################
####  DATASET

# Apriamo il dataset (supponiamo di aver già montato gdrive su colab)
datadir = datadir + 'Regression_&_Classification/'
file_path = datadir +'Linear_Regression_Single_Carburante.csv'
data = np.loadtxt(file_path, delimiter=',', skiprows=1)
x = data[:, 0]  ## Velocità (Km/h)
y = data[:, 1]  ## Consumo (l/100Km)

# Plottiamo il dataset
plt.scatter(x, y, color='red', marker='x')
plt.xlabel('Velocità (km/h)'); plt.ylabel('Consumo (litri/100 km)')
plt.legend(); plt.title('Consumo di carburante vs Velocità'); plt.show()


#############################################################################################
#############################################################################################
####  MODEL IMPLEMENTATION

# Train-Test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, 1); X_test = X_test.reshape(-1, 1)

# Definire e addestrare il modello
model = LinearRegression()
model.fit(X, y)

# Ottenimento dei coefficienti del modello
w = model.coef_[0]; b = model.intercept_
print(f"w: {w}, b: {b}")

# Per fare le previsioni
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Visualizzazione dei risultati
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='red', marker='x', label='Training Data')
plt.scatter(X_test, y_test, color='green', marker='o', label='Test Data')
plt.plot(X_train, y_pred_train, color='blue', label='Prediction on Training Data')
plt.plot(X_test, y_pred_test, color='orange', linestyle='--', label='Prediction on Test Data')
plt.xlabel('Velocità (km/h)'); plt.ylabel('Consumo (litri/100 km)'); plt.title('Consumo di carburante vs Velocità')
plt.legend(); plt.show()









 
