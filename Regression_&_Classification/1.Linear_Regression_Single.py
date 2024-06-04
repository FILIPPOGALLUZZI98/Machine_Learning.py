import numpy as np
import matplotlib.pyplot as plt
import math, copy


#############################################################################################
#############################################################################################

# Il modello sarà del tipo f_wb = w_x + b

# Consideriamo i seguenti dati
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,]) 

# Facciamo il grafico per vedere che forma hanno i dati
plt.scatter(x_train, y_train, marker='x', c='r')
plt.show()

# Costruiamo ora la funzione del modello con valori di w e b scelti a caso (per ora)
w = 100; b = 100

def model_single(x, w, b):
    m = x.shape[0]  ## [0] indica la dimensione del dataset, ovvero il # di training examples
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b      
    return f_wb

# Per applicare il modello
tmp_f_wb = model(x_train, w, b,)
# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.show()

#############################################################################################
#############################################################################################
#### GRADIENT DESCENT ALGORITHM

# Funzione per calcolare la Cost Function
def cost_single(x, y, w, b): 
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
def gradient_descent_single(x, y, w_in, b_in, alpha, num_iters, cost_single, gradient_single): 
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_single(x, y, w , b)     
        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_single(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J_history, p_history #return w and J,w history for graphing


# Per applicare il GD è necessario impostare il punto di partenza e altre variabili
w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2
# Run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent_single(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, cost, gradient)













