# Softmax Regression is a generalization of logistic regression algorithm

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# In questo caso usiamo l'esempio di handwritten digit recognition (quindi con 
# l'output = 10 cifre)

# Costruire il modello di rete neurale
model = Sequential([
  Dense(units=25, activation='relu'),
  Dense(units=15, activation='relu'),
  Dense(units=10, activation='softmax'))]

# Specificare loss and cost
model.compile(loss= SparseCategoricalCrossentropy())

# Train on data to minimize the loss function
model.fit(X,Y, epochs=100)






