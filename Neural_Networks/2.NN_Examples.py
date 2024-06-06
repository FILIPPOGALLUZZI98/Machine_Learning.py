# Codice per implementare una rete neurale semplice
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid

#############################################################################################
#############################################################################################
####  HANDWRITTEN DIGIT RECOGNITION: 0-1 (vedi appunti quaderno)

# Questo codice è solo per dimostrazione, non funziona 
# NN con 3 layer: 25, 15, 1
x = np.array([[0..0, ..., 245, ..., 12, 0]])  ## I valori sono le intensità dei vari pixel
                                              ##  che compongono l'immagine 
layer_1 = Dense(units=25, activation='sigmoid')  ## Primo layer con 25 neuroni
a1 = layer_1(x)
layer_2 = Dense(units=15, activation='sigmoid')  ## Secondo layer con 15 neuroni
a2 = layer_2(a1)
layer_3 = Dense(units=1, activation='sigmoid')  ## Terzo layer (output) con 1 neurone
a3 = layer_3(a2)

# Per fare la previsione del numero
if a3 >= 0.5:
  yhat = 1
else:
  yhat = 0


# Per rendere il modello più compatto e veloce
model = Sequential([
        tf.keras.Input(shape=(2,)),
        Dense(25, activation='sigmoid', name = 'layer1'),
        Dense(15, activation='sigmoid', name = 'layer2'),
        Dense(1, activattion='sigmoid', name = 'output')])

# Per conoscere i pesi e i bias
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()

#############################################################################################
#############################################################################################
####  ROASTING COFFEE: GOOD OR BAD (Temperature and Time)

# Roasting Coffee: 2 layer con neuroni = 3, 1
# I dati in input sono X = (Temperatura, Tempo) e Y = (Buono = 1, Non Buono = 0)
# Usiamo come input
X = np.array([[185.32,  12.69],  [259.92,  11.87],  [231.01 , 14.41],  [175.37 , 11.72],  [187.12 , 14.13],
     [225.91 , 12.1 ], [208.41 , 14.18],  [207.08 , 14.03],  [280.6  , 14.23],  [202.87 , 12.25],
     [196.7, 13.54], [270.31, 14.6], [192.95, 15.2], [213.57, 14.28], [164.47, 11.92],
     [177.26, 15.04], [241.77, 14.9], [237.0, 13.13], [219.74, 13.87], [266.39, 13.25],
     [270.45, 13.95], [261.96, 13.49], [243.49, 12.86], [220.58, 12.36], [163.59, 11.65],
     [244.76, 13.33], [271.19, 14.84], [201.99, 15.39]])
Y = np.array([[1.],  [0.],  [0.],  [0.],  [1.],  [1.],  [0.],  [0.],  [0.],  [1.], 
             [1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [0.], [0.], [0.], [0.]])
for i in range(len(Y)):
    if Y[i] == 0:
        plt.scatter(X[i, 0], X[i, 1], color='blue', label='Class 0' if i == 0 else "")
    else:
        plt.scatter(X[i, 0], X[i, 1], color='red', label='Class 1' if i == 1 else "")

# Normalizziamo i dati per rendere la futura backpropagation più veloce
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
# Tile/copy our data to increase the training set size and reduce the number of training epochs
Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1))   
print(Xt.shape, Yt.shape) 

# A questo punto possiamo costruire il modello
tf.random.set_seed(1234)  ## Per avere lo stesso valore se ripetuto 
model = Sequential([
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')])

# Per conoscere i pesi e i bias
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()


# The model.compile statement defines a loss function and specifies a compile optimization.
# The model.fit statement runs gradient descent and fits the weights to the data.
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),)
model.fit(
    Xt,Yt,            
    epochs=10,)
# In the fit statement above, the number of epochs was set to 10. This specifies that the entire
# data set should be applied during training 10 times.
# Vediamo i nuovi pesi e bias dopo il training
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()

# Per testare il modello
X_test = np.array([
    [200,13.9],  # positive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)

# Per la decisione
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")









