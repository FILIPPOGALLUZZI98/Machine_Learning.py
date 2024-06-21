# Supponiamo di dover prevedere prevedere il consumo di carburante (in litri per 100 km)
# di un'auto in base alla sua velocità media (in km/h).
# Si tratta di una regressione lineare con singola variabile
# Il modello sarà del tipo f_wb = w*x + b
# Per una singola variabile non c'è il la regularization perché l'overfitting non è un problema

import numpy as np
import matplotlib.pyplot as plt
import math, copy


#############################################################################################
#############################################################################################
####  DATASET

# Apriamo il dataset (supponiamo di aver già montato gdrive su colab)
datadir = datadir + 'Regression_&_Classification/'
file_path = datadir +'Linear_Regression_Single_Carburante.csv'
data = np.loadtxt(file_path, delimiter=',', skiprows=1)
x = data[:, 0]  ## Velocità (Km/h)
y = data[:, 1]  ## Consumo (l/100Km)

# Plottiamo il dataset
plt.scatter(x, y, color='red', marker='x')
plt.xlabel('Velocità (km/h)'); plt.ylabel('Consumo (litri/100 km)')
plt.legend(); plt.title('Consumo di carburante vs Velocità');plt.show()


#############################################################################################
#############################################################################################
####  MODELLO

# La funzione che ci fornisce la previsione (cioè il modello)
def model_single(x, w, b):
    m = x.shape[0]  ## [0] indica la dimensione del dataset, ovvero il numero di training examples
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b      
    return f_wb
  
# Per fare la prova, basta inserire, per adesso, valori di w e b casuali
w = -0.05; b = 8
f_wb = model_single(x, w, b)

# Plot our model prediction and the overlapped data points
plt.plot(x, f_wb, c='b',label='Prediction')
plt.scatter(x, y, marker='x', c='r',label='Actual Values'); plt.legend();plt.show()


#############################################################################################
#############################################################################################
####  GRADIENT DESCENT ALGORITHM (DEFINIZIONI DELLE FUNZIONI)

# Definiamo le funzioni 'cost_function_single', 'gradient_single' e 'gradient_descent_single'

# Cost function
def cost_function_single(x, y, w, b): 
    m = x.shape[0] 
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        temp_cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + temp_cost  
    cost = (1 / (2 * m)) * cost_sum  
    return cost

# Funzione per la derivata (gradiente)
def gradient_single(x, y, w, b): 
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
    return dj_dw, dj_db

# Funzione per il Gradient Descent (GD) con storia della funzione di costo (ognmi 100 iterazioni)
def gradient_descent_single(x, y, w_in, b_in, alpha, num_iters, cost_function_single, gradient_single): 
    b = b_in
    w = w_in
    cost_history = []
    for i in range(num_iters):
        dj_dw, dj_db = gradient_single(x, y, w, b)     
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            
        if i % 100 == 0 or i == num_iters - 1:  
            cost = cost_function_single(x, y, w, b)
            cost_history.append(cost)
    return w, b, cost_history


#############################################################################################
#############################################################################################
####  APPLICARE GRADIENT DESCENT 

# Impostiamo il punto di partenza e altre variabili
w_init = 0; b_init = 0; iterations = 1000000; tmp_alpha = 1.0e-4

# Eseguire gradient descent
w_final, b_final, cost_history = gradient_descent_single(x, y, w_init, b_init, tmp_alpha, iterations, cost_function_single, gradient_single)

# Plot della funzione di costo in funzione delle iterazioni
plt.plot(range(len(cost_history)), cost_history, 'b')
plt.xlabel('Iterazioni (x100)'); plt.ylabel('Costo'); plt.title('Andamento del Costo durante il Gradient Descent')
plt.show()

# A questo punto possiamo fare qualche prova cambiando alpha e iterations per vedere i valori migliori
# Per quei valori allora cerchiamo w e b
print(f"w_final: {w_final}"); print(f"b_final: {b_final}")

# A questo punto, per vedere la retta trovata
f_wb = model_single(x, w_final, b_final)
plt.plot(x, f_wb, c='b',label='Our Prediction')
plt.scatter(x, y, marker='x', c='r',label='Actual Values')
plt.show()












