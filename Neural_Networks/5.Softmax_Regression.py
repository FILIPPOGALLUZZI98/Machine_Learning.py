# Softmax Regression is a generalization of logistic regression algorithm

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy


# Il seguente è il codice per fare degli esempi casuali di dati cluster attorno a 4 centri
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)


#############################################################################################
#############################################################################################
####  MODEL IMPLEMENTATION (la versione consigliata invece è quella dopo)

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


#############################################################################################
#############################################################################################
####  MODEL IMPLEMENTATION (NUMERICALLY IMPROVED)

# Il problema è lo stesso, ma se vogliamo rendere l'algoritmo più accurato (cioè ridurre lo 
# spazio di memoria di lavoro e quindi renderlo più veloce e ridurre i numerical round-off errors)

# Costruire il modello di rete neurale (con output 'linear'
model = Sequential([
  Dense(units=25, activation='relu'),
  Dense(units=15, activation='relu'),
  Dense(units=10, activation='linear'))]

# Specificare loss and cost
model.compile(loss= SparseCategoricalCrossentropy(from_logits=True))

# Train on data to minimize the loss function
model.fit(X,Y, epochs=100)

# Per fare la previsione
logit = model(X)
f_x = ft.nn.sigmoid(logit)


#############################################################################################
#############################################################################################
####  SOFTMAX FUNCTION IMPLEMENTATION

def my_softmax(z):
    ez = np.exp(z) 
    sm = ez/np.sum(ez)
    return(sm)















