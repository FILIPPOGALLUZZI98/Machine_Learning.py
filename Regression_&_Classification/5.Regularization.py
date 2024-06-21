import numpy as np
import matplotlib.pyplot as plt
import math, copy

# Codici che usano variabili multiple: usando vettorizzazione
# Questo file contiene i codici per la cost function e la funzione gradiente regolarizzate
# sia per la linear che la logistic regression
# La funzione del GD rimane invariata a quella dei file '2.Linear_Regression_Multiple.py' e
# '4.Logistic_Regression.py'


#############################################################################################
#############################################################################################
####  LOGISTIC REGRESSION

# Algorithm for computing the regularized cost
def cost_regularized_logistic(X, y, w, b, lambda_ = 1):
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
def gradient_regularized_logistic(X, y, w, b, lambda_): 
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



