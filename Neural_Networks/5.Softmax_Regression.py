# Softmax Regression is a generalization of logistic regression algorithm

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy


#############################################################################################
#############################################################################################
####  MODEL IMPLEMENTATION

# In questo caso usiamo l'esempio di handwritten digit recognition (quindi con 
# l'output = 10 cifre)

# Costruire il modello di rete neurale
model = Sequential([
  Dense(units=25, activation='relu'),
  Dense(units=15, activation='relu'),
  Dense(units=10, activation='softmax'))]

# Specificare loss and cost
model.compile(loss= SparseCategoricalCrossentropy())
## Se vogliamo rendere l'algoritmo più accurato (cioè ridurre lo spazio di memoria di lavoro
## e quindi renderlo più veloce e ridurre i numerical round-off errors):
# model.compile(loss= SparseCategoricalCrossentropy(from_logits=True))

# Train on data to minimize the loss function
model.fit(X,Y, epochs=100)
## Se usiamo il modello improved allora:
# logit = model(X)
# f_x = ft.nn.sigmoid(logit)  ## Restituisce le z_j




















