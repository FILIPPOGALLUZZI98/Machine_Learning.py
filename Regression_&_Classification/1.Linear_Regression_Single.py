import numpy as np
import matplotlib.pyplot as plt
import math, copy

# Algoritmo per trovare i valori dei pesi e dei bias per la regressione lineare che meglio
# si adattano ai dati
# Il modello sarà del tipo f_wb = w*x + b

# Consideriamo i seguenti dati
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,]) 


#############################################################################################
#############################################################################################
####  MODELLO E PLOT DEI DATI

# Facciamo il grafico per vedere che forma hanno i dati
plt.scatter(x_train, y_train, marker='x', c='r'); plt.show()

# Costruiamo ora la funzione che restituisce f_wb=yhat, ovvero la previsione
def model_single(x, w, b):
    m = x.shape[0]  ## [0] indica la dimensione del dataset, ovvero il # di training examples
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b      
    return f_wb

# Per applicare il modello, scegliamo (a caso, per ora) i valori dei pesi e del bias
w = 100; b = 100
tmp_f_wb = model_single(x_train, w, b)
# Plot our model prediction and the overlapped data points
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.show()


#############################################################################################
#############################################################################################
####  GRADIENT DESCENT ALGORITHM (per trovare w, b)

# Funzione per calcolare la Cost Function
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

# Funzione per il Gradient Descent (GD)
def gradient_descent_single(x, y, w_in, b_in, alpha, num_iters, cost_function_single, gradient_single): 
    b = b_in
    w = w_in
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_single(x, y, w , b)     
        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            
    return w, b 

# Per applicare il GD è necessario impostare il punto di partenza e altre variabili
w_init = 0; b_init = 0; iterations = 10000; tmp_alpha = 1.0e-2
# Run gradient descent
w_final, b_final = gradient_descent_single(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                           iterations, cost_function_single, gradient_single)
print(w_final); print(b_final)

# A questo punto, per vedere la retta trovata
f_wb = model_single(x_train, w_final, b_final)
plt.plot(x_train, f_wb, c='b',label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.show()











