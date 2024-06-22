import copy, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Supponiamo di dover prevedere il prezzo di una casa (variabile dipendente) 
# Questo può essere influenzato da fattori come la dimensione, il numero di stanze, l'età della casa, la vicinanza dal centro
# Si tratta di una regressione lineare con variabili multiple
# Il modello sarà del tipo f_wb = w1*x1 + w2*x2 + ... + b


#############################################################################################
#############################################################################################
####  DATASET

# Apriamo il dataset (supponiamo di aver già montato gdrive su colab)
datadir = datadir + 'Regression_&_Classification/'
file_path = datadir +'Linear_Regression_Multiple_House.csv'
df = pd.read_csv(file_path)

# Plottiamo i 4 grafici
features = ['Dimensione (m^2)', 'Numero di stanze', 'Età della casa (anni)', 'Distanza dal centro (km)']
titles = ['Dimensione vs Prezzo', 'Numero di stanze vs Prezzo', 'Età della casa vs Prezzo', 'Distanza dal centro vs Prezzo']
plt.figure(figsize=(12, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i + 1)
    plt.scatter(df[feature], df['Prezzo (€)'], s = 1, color = 'k')
    plt.title(titles[i]); plt.xlabel(feature); plt.ylabel('Prezzo (€)')
plt.tight_layout(); plt.show()


#############################################################################################
#############################################################################################
####  MODEL IMPLEMENTATION

# Selezionare le variabili del dataset
X = df[features].values; y = df['Prezzo (€)'].values

# Dividere i dati in train set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creazione e addestramento del modello
model = LinearRegression()
model.fit(X_train, y_train)

# Ottenere i parametri del modello
w = model.coef_; b = model.intercept_
print(f"w: {w}, b: {b}")

# Fare previsioni su un nuovo esempio
x_vec = np.array([[120, 5, 20, 40]])
f_wb = model.predict(x_vec)
print(f"Predicted price for {x_vec}: {f_wb}")

# Previsioni sul test set
y_pred_test = model.predict(X_test)

# Visualizzazione delle prestazioni del modello per ciascuna variabile
plt.figure(figsize=(12, 12))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i + 1)
    plt.scatter(df[feature], df['Prezzo (€)'], s=1, color='k', label='Dati')
    x_vals = np.linspace(df[feature].min(), df[feature].max(), 100).reshape(-1, 1)
    X_vals = np.full((x_vals.shape[0], X.shape[1]), df[features].mean(axis=0))
    X_vals[:, i] = x_vals.flatten()
    y_vals = model.predict(X_vals)
    plt.plot(x_vals, y_vals, color='r', label='Regressione'); plt.title(titles[i])
    plt.xlabel(feature); plt.ylabel('Prezzo (€)'); plt.legend()
plt.tight_layout(); plt.show()

# Calcolo dell'MSE sul test set
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Coefficiente di determinazione R^2: {r2}")

# Plot dei residui
residui = y_test - y_pred_test
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_test, residui, color='blue')
plt.xlabel('Previsioni'); plt.ylabel('Residui'); plt.axhline(y=0, color='r', linestyle='--')
plt.title('Plot dei Residui'); plt.grid(True); plt.show()











