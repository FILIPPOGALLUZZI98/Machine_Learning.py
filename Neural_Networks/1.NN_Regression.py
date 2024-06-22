import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# In questo esempio vogliamo predire il valore di una variabile y che può assumere
# valori continui positivi e negativi che dipende da tre variabili X1, X2, X3
# Dato che è un problema di regressione i modelli usati avranno HL = relu e OL = linear


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
####  MODEL SELECTION

# Creiamo i vettori dei dati
X = df[['X1', 'X2', 'X3']].values
y = df['Y'].values

# Splittiamo i dati in 2 insiemi (train e test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80-20 split

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
    {'layers': 3, 'units': [10, 5, 10], 'activation': ['relu', 'relu', 'relu']}]

# Cross-validation con KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

val_loss_dict = {i: [] for i in range(len(models_config))}
# Addestramento dei modelli e raccolta delle performance
for train_index, val_index in kf.split(X_train):
    X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
    y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
    for i, config in enumerate(models_config):
        model = create_model(config['layers'], config['units'], config['activation'])
        history = model.fit(X_train_cv, y_train_cv, epochs=100, batch_size=32, validation_data=(X_val_cv, y_val_cv), verbose=0)
        val_loss_dict[i].append(history.history['val_loss'][-1])
# Calcola la media della validation loss per ogni modello
avg_val_loss_list = [np.mean(val_loss_dict[i]) for i in range(len(models_config))]
# Seleziona il miglior modello basato sulla media della validation loss
best_model_index = np.argmin(avg_val_loss_list)
best_model_config = models_config[best_model_index]

# Visualizzazione delle performance
plt.figure(figsize=(10, 6))
for i in range(len(models_config)):
    plt.plot(val_loss_dict[i], label=f'Model {i+1}')
plt.xlabel('Fold'); plt.ylabel('Validation Loss'); plt.legend()
plt.title('Confronto della Loss su Validation Set tra i Modelli'); plt.show()

# Stampa della performance finale di ogni modello
for i, avg_val_loss in enumerate(avg_val_loss_list):
    print(f'Model {i+1} - Average Validation Loss: {avg_val_loss}')

# Addestramento del modello migliore su tutto il training set e valutazione sul test set
best_model = create_model(best_model_config['layers'], best_model_config['units'], best_model_config['activation'])
best_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
y_test_pred = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Miglior modello - Test MSE: {test_mse}")


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






