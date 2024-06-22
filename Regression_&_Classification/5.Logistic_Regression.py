import numpy as np
import matplotlib.pyplot as plt
import math, copy

# La logistic regression viene usata nei problemi di classificazione (binaria)
# e anche come funzione di attivazione nelle reti neurali
# Codice per logistic regression con più variabili: usando vettorizzazione

# Implementation of sigmoid function:
def sigmoid(z):
    g = 1/(1+np.exp(-z))   
    return g

# Cost Function for Logistic Regression
def cost_logistic(X, y, w, b, lambda_ = 1):
    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b                                      
        f_wb_i = sigmoid(z_i)                                         
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)     
    cost = cost/m                                                     
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                         
    reg_cost = (lambda_/(2*m)) * reg_cost                                 
    total_cost = cost + reg_cost                                       
    return total_cost    

# Algorithm for computing the regularized gradient 
def gradient_logistic(X, y, w, b, lambda_): 
    m,n = X.shape
    dj_dw = np.zeros((n,))                            
    dj_db = 0.0                                       
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          
        err_i  = f_wb_i  - y[i]                       
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   
    dj_db = dj_db/m                                   
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]
    return dj_db, dj_dw 

# Gradient Descent
def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_logistic(X, y, w, b)   
        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
    return w, b   


# Per fare prova
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
w_tmp  = np.zeros_like(X_train[0]); b_tmp  = 0.; alph = 0.1; iters = 10000
w_out, b_out = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(w_out); print(b_out)

