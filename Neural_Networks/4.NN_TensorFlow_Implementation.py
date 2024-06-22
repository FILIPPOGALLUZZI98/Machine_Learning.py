import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

# Come esempio prendiamo il riconoscimento di cifre (0 o 1) come visto negli appunti
# layer1 = 25n; layer2 = 15n; layer3 = output = 1n


# Costruiamo il modello di rete neurale
model = Sequential([
  Dense(units=25, activation='sigmoid'),
  Dense(units=15, activation='sigmoid'),
  Dense(units=1, activation='sigmoid')])

# Specifichiamo quale loss function vogliamo usare
model.compile(loss=Binarycrossentropy())

# Fit the model (minimizzare la loss function)
model.fit(X,Y,epochs=100)  ## How many steps for gradient descent






















