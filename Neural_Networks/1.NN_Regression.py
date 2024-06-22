import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# In questo esempio vogliamo predire il valore di una variabile y che può assumere
# valori continui positivi e negativi che dipende da tre variabili X1, X2, X3


#############################################################################################
#############################################################################################
####  DATASET

# Apriamo il dataset (supponiamo di aver già montato gdrive su colab)
datadir = datadir + 'Neural_Networks/'
file_path = datadir +'Esempio_1.csv'
df = pd.read_csv(file_path)

# Plottiamo i grafici delle variabili
features = ['X1','X2', 'X3']  ## Sono i nomi delle colonne del dataset
titles = ['Variabile 1', 'Variabile 2','Variabile 3']
plt.figure(figsize=(12,10))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i + 1)
    plt.scatter(df[feature], df['Y'], s = 1, color = 'k')
    plt.title(titles[i]); plt.ylabel('Variabile Y')
plt.tight_layout(); plt.show()


#############################################################################################
#############################################################################################
####  MODELLO

# Creiamo i vettori dei dati
X = df[['X1', 'X2', 'X3']].values
y = df['Y'].values 

# Splittiamo i dati in 3 insiemi (train, cross-validation e test) 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)  ## 80-20
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  ## 50-50

# Creazione del modello
model = Sequential([
Dense(units = 20, input_shape=(X_train.shape[1],), activation='relu'),
Dense(units = 10, activation='relu'),
Dense(units = 1,  activation='linear')])
# Negli hidden layer si usa relu perché è una regressione, e nell'output
# layer si usa linear activation function

# Compilazione del modello
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# Ottimizzazione = Adam
# Funzione di perdita dell'errore quadratico medio (MSE); # Binarycrossentropy() 
# Metrica MAE (Mean Absolute Error) per monitorare le prestazioni

# Addestramento del modello
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)
# epochs = numero di iterazioni
# batch_size = numero di campioni da utilizzare per aggiornare i pesi del modello durante l'addestramento

# Valutazione del modello sul set di test
loss, mae = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, MAE: {mae}')

# Previsione con il modello
predictions = model.predict(X_test)
for i in range(5):
    print(f'Valore Reale: {y_test[i]}, Valore Predetto: {predictions[i]}')

for layer in model.layers:
    weights, biases = layer.get_weights()
    print(f'Layer: {layer.name}')
    print(f'  Weights: {weights}')
    print(f'  Biases: {biases}')


#############################################################################################
#############################################################################################
####  MODEL SELECTION

# Creiamo i vettori dei dati
X = df[['X1', 'X2', 'X3']].values
y = df['Y'].values 

# Splittiamo i dati in 3 insiemi (train, cross-validation e test) 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  ## 80-20
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  ## 50-50


# Funzione per creare modelli
def create_model(layers, units, activation):
    model = Sequential()
    model.add(Dense(units=units[0], input_shape=(X_train.shape[1],), activation=activation[0]))
    for i in range(1, layers):
        model.add(Dense(units=units[i], activation=activation[i]))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Definizione delle architetture dei modelli
models_config = [
    {'layers': 2, 'units': [20, 10], 'activation': ['relu', 'relu']},
    {'layers': 2, 'units': [15, 10], 'activation': ['relu', 'relu']},
    {'layers': 3, 'units': [30, 20, 10], 'activation': ['relu', 'relu', 'relu']},
    {'layers': 3, 'units': [10,5, 10], 'activation': ['relu', 'relu','relu']}
]

# Addestramento dei modelli e raccolta delle performance
history_list = []; val_loss_list = []
for config in models_config:
    model = create_model(config['layers'], config['units'], config['activation'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    history_list.append(history)
    val_loss_list.append(history.history['val_loss'][-1])

# Visualizzazione delle performance
plt.figure(figsize=(10, 6))
for i, history in enumerate(history_list):
    plt.plot(history.history['val_loss'], label=f'Model {i+1}')
plt.xlabel('Epochs'); plt.ylabel('Validation Loss'); plt.legend()
plt.title('Confronto della Loss su Validation Set tra i Modelli'); plt.show()

# Stampa della performance finale di ogni modello
for i, val_loss in enumerate(val_loss_list):
    print(f'Model {i+1} - Validation Loss: {val_loss}')




