import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Importare la sigmoid function

# Codice per implementare una rete neurale senza pytorch ma direttamente da numpy

x = np.array([200, 17])
W = np.array([[1,-3,5],
              [-2,4,-6)]])
b = np.array([-1,1,2])

# For activation of a single layer
def dense(a_in, W,b):
    units = W.shape[1]
    a_out = np.zeros(units):
    for j in range(units):
        w = W[:,j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out

# Funzione per rete neurale
def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x,  W1, b1)
    a2 = my_dense(a1, W2, b2)
    return(a2)

# Funzione per fare una previsione
def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return(p)

# Fare la previsione con dati esempio
X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_tstn = norm_l(X_tst)  ## Normalizzazione con il pacchetto pytorch
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")


#############################################################################################
#############################################################################################
####  VECTORIZATION

# Usando la vettorizzazione (ricordando che i vettori sono diversi in pytorch e numpy
# Gli algoritmi sono più efficenti e veloci

X = np.array([[200,17]])
W = np.array([[1,-3,5],
              [-2,4,-6)]])
b = np.array([[-1,1,2]])

# Per attivazione di un singolo layer
def dense(A_in, W,B):
    Z = np.matmul(A_in, W) + B  ## Matrix multiplication
    A_out = g(Z)
    return A_out








