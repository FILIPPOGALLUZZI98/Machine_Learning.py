# Questo viene usato quando l'output y ha più di un valore (quindi per classificare più
# di due output

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy


# Il seguente è il codice per fare degli esempi casuali di dati cluster attorno a 4 centri
import sklearn
from sklearn.datasets import make_blobs
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)


#############################################################################################
#############################################################################################
####  MODEL IMPLEMENTATION 

# Costruzione del modello
tf.random.set_seed(1234)  ## Per ottere gli stessi risultati
model = Sequential([
        Dense(2, activation = 'relu',   name = "L1"),
        Dense(4, activation = 'linear', name = "L2")])

# Compile the network
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),)

# Model fit
model.fit(X_train, y_train, epochs=200)





