# Come esempio sarà usata la previsione del prezzo della casa
# Ci saranno molte variabili tra cui: area, numero di piani, numero di stanze, etc.


import numpy as np
import matplotlib.pyplot as plt





#############################################################################################
#### MODELLO
# Il modello sarà del tipo f_wb = w_1*x_1 + w_2*x_2 + ... + b

# Supponiamo di avere i seguenti dati sull'area (x) e sul prezzo (y)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# Chiameremo con m il numero di training examples
m = x_train.shape[0]  ## [0] indica la dimensione di x di cui vogliamo sapere la lunghezza

# Possiamo anche usare la forma generale
i = 0 
x_i = x_train[i]
y_i = y_train[i]

# Facciamo il grafico per vedere che forma hanno
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.show()

# Costruiamo ora la funzione del modello con valori di w e b scelti a caso (per ora)
w = 100
b = 100

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b      
    return f_wb

# Per vedere se funziona
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
















