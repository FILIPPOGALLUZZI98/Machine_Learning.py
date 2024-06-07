# Alcuni pacchetti utili
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid


#############################################################################################
#############################################################################################
####  NEURON WITHOUT ACTIVATION 

# Usiamo i dati seguenti
X_train = np.array([[1.0], [2.0]], dtype=np.float32)   
Y_train = np.array([[300.0], [500.0]], dtype=np.float32) 
fig, ax = plt.subplots(1,1); ax.scatter(X_train, Y_train); plt.show()

# Definiamo un layer con un solo neurone e lo compariamo con la regressione lineare
linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )
# Inserendo il vettore X_train allora i pesi vengono inizializzati (con valori w piccoli e b=0)
a1 = linear_layer(X_train[0].reshape(1,1))  ## Funzione di attivazione
print(a1)
w, b= linear_layer.get_weights()  ## Per vedere i valori di w e b
print(f"w = {w}, b={b}")

# Per settare w e b a valori scelti
set_w = np.array([[200]]); set_b = np.array([100])
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())

# Se vogliamo fare una previsione
prediction_tf = linear_layer(X_train)  ## Questa è la previsione con il layer
prediction_np = np.dot( X_train, set_w) + set_b  ## Questo è manuale!
plt_linear(X_train, Y_train, prediction_tf, prediction_np)  ## In questo caso è lo stesso


#############################################################################################
#############################################################################################
####  NEURON WITH SIGMOID ACTIVATION

# Usiamo il seguente dataset
X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)
pos = Y_train == 1; neg = Y_train == 0
fig,ax = plt.subplots(1,1,figsize=(4,3))
ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c = 'red', label="y=1")
ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", c='blue',lw=1)

# Creare un logistic neuron
model = Sequential([tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')])

# Per vedere le caratteristiche
model.summary()
logistic_layer = model.get_layer('L1')
w,b = logistic_layer.get_weights()
print(w,b)
print(w.shape,b.shape)

# Impostare pesi e bias con valori scelti
set_w = np.array([[2]]); set_b = np.array([-4.5])
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())

# Per fare previsioni
plt_logistic(X_train, Y_train, model, set_w, set_b, pos, neg)






