#############################################################################################
#############################################################################################
####  LINEAR REGRESSION (SINGLE VARIABLE)

import numpy as np
import matplotlib.pyplot as plt
import math, copy

# Supponiamo di dover prevedere prevedere il consumo di carburante (in litri per 100 km)
# di un'auto in base alla sua velocità media (in km/h).
# Si tratta di una regressione lineare con singola variabile
# Il modello sarà del tipo f_wb = w*x + b
# Per una singola variabile non c'è il la regularization perché l'overfitting non è un problema


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
plt.legend(); plt.title('Consumo di carburante vs Velocità'); plt.show()


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




#############################################################################################
#############################################################################################
####  LINEAR REGRESSION (MULTIPLE VARIABLES)

import copy, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Supponiamo di dover prevedere il prezzo di una casa (variabile dipendente) 
# Questo può essere influenzato da fattori come la dimensione, il numero di stanze, l'età della casa, la vicinanza dal centro
# Si tratta di una regressione lineare con variabili multiple
# Il modello sarà del tipo f_wb = w1*x1 + w2*x2 + ... + b



####  DATASET

# Apriamo il dataset (supponiamo di aver già montato gdrive su colab)
datadir = datadir + 'Regression_&_Classification/'
file_path = datadir +'Linear_Regression_Multiple_House.csv'
df = pd.read_csv(file_path)

# Plottiamo i 4 grafici
features = ['Dimensione (m^2)', 'Numero di stanze', 'Età della casa (anni)', 'Distanza dal centro (km)']
titles = ['Dimensione vs Prezzo', 'Numero di stanze vs Prezzo', 'Età della casa vs Prezzo', 'Distanza dal centro vs Prezzo']
plt.figure(figsize=(12, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i + 1)
    plt.scatter(df[feature], df['Prezzo (€)'], s = 1, color = 'k')
    plt.title(titles[i]); plt.xlabel(feature); plt.ylabel('Prezzo (€)')
plt.tight_layout(); plt.show()


####  MODELLO

# Creiamo due vettori con le variabili di interesse
X = df[['Dimensione (m^2)', 'Numero di stanze', 'Età della casa (anni)', 'Distanza dal centro (km)']].values
y = df['Prezzo (€)'].values

# La funzione che ci fornisce la previsione (cioè il modello)
def model(x, w, b): 
    p = np.dot(x, w) + b     
    return p  
  
# Per fare una previsione, selezioniamo un nuovo vettore esempio
x_vec = np.array([[120, 5, 20, 40]])
w_init =np.array([3020, 50000, -2080, -1100]); b_init= -1400
f_wb = model(x_vec, w_init, b_init)


####  GRADIENT DESCENT ALGORITHM

# Per fare la cost function
def cost_function(X, y, w, b): 
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b    
        cost_temp = cost + (f_wb_i - y[i])**2     
    cost = cost_temp / (2 * m)                      
    return cost

# Per fare il gradiente in più dimensioni
def gradient(X, y, w, b): 
    m,n = X.shape  
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_db, dj_dw = gradient(X, y, w, b)   
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append(cost_function(X, y, w, b) )
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    return w, b, J_history         


####  APPLICARE GRADIENT DESCENT 

# Facciamo i grafici per 4 valori di alpha
initial_w = np.zeros(X.shape[1]); initial_b = 0.; iterations = 10000

alpha_values = [1e-7, 1e-6, 1e-5, 1e-4]
plt.figure(figsize=(14, 10))
for i, alpha in enumerate(alpha_values):
    _,_,J_history = gradient_descent(X, y, initial_w, initial_b, alpha=alpha, num_iters=iterations)
    plt.subplot(2, 2, i + 1)
    plt.plot(J_history); plt.title(f'Alpha = {alpha}'); plt.xlabel('Iterations'); plt.ylabel('Cost J')
plt.tight_layout(); plt.show()

# Facciamo ora il gradient descent con il migliore valore di alpha
initial_w = np.zeros(X.shape[1]); initial_b = 0.; iterations = 1000000; alpha = 1.0e-4
w_final, b_final, _ = gradient_descent(X, y, initial_w, initial_b, alpha, iterations)

# Facciamo i 4 grafici delle possibili variabili per vedere come influiscono sul prezzo
plt.figure(figsize=(10, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i + 1)
    plt.scatter(df[feature], df['Prezzo (€)'], s=1, color='k', label='Dati')
    x_vals = np.linspace(df[feature].min(), df[feature].max(), 100).reshape(-1, 1)
    X_vals = np.full((x_vals.shape[0], X.shape[1]), df[features].mean(axis=0))
    X_vals[:, i] = x_vals.flatten()
    y_vals = model(X_vals, w_final, b_final)
    plt.plot(x_vals, y_vals, color='r', label='Regressione'); plt.title(titles[i])
    plt.xlabel(feature); plt.ylabel('Prezzo (€)'); plt.legend()
plt.tight_layout(); plt.show()

# Per fare una previsione del prezzo 
x_vec = np.array([[120, 5, 20, 40]])
f_wb = model(x_vec, w_final, b_final); f_wb




#############################################################################################
#############################################################################################
####  POLYNOMIAL REGRESSION

import copy, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Il modello sarà del tipo f_wb = w1*x1 + w2*x1^2 + w3*x2^(-1/2) + w4*x3^3 + ... + b
# In questo caso dobbiamo usare la regolarizzazione per evitare l'overfitting

# Supponiamo di dover prevedere la resa di un campo in base ad alcune variabili come
# la quantità di fertilizzanti usati, di acqua e ore di sole
# Usiamo solo queste 3 variabili (anche se nel dataset ce ne sono di più)


####  DATASET

# Apriamo il dataset (supponiamo di aver già montato gdrive su colab)
datadir = datadir + 'Regression_&_Classification/'
file_path = datadir +'Polynomial_Regression_Multiple_Crop.csv'
df = pd.read_csv(file_path)

# Plottiamo i 3 grafici (fertilizer, water, sunlight vs. crop yield)
features = ['fertilizer_used','water_irrigated', 'sunlight_hours']
titles = ['Fertilzer', 'Water','Sunlight']
plt.figure(figsize=(12,10))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i + 1)
    plt.scatter(df[feature], df['crop_yield'], s = 1, color = 'k')
    plt.title(titles[i]); plt.ylabel('Crop Yield')
plt.tight_layout(); plt.show()


####  MODELLO

# Supponiamo di volere il modello: f_wb = w1*x1 + w2*x1^3 + w3*x2 + w4*x2^2 + w5*x3 + w6*x3^2 + b
# x1 -> fertilizer; x2 -> water; w3 -> sunlight

# Costruiamo due engineered variables per ottenere: X = (x1, x1^3, x2, x2^2, x3, x3^2)
# In questo caso creiamo 3 colonne aggiuntive al dataset df
df['fertilizer_used_sq'] = df['fertilizer_used'] ** 3
df['water_irrigated_sq'] = df['water_irrigated'] ** 2
df['sunlight_hours_sq'] = df['sunlight_hours'] ** 2

# Creiamo il vettore X con le colonne desiderate
X = df[['fertilizer_used', 'fertilizer_used_sq', 'water_irrigated', 'water_irrigated_sq',
        'sunlight_hours', 'sunlight_hours_sq']].values
y = df['crop_yield'].values  ## Variabile dipendente


# Il modello sarà quindi lo stesso che per la regressione lineare con più variabili
def model(x, w, b): 
    p = np.dot(x, w) + b     
    return p 

# Per fare una previsione, selezioniamo un nuovo vettore esempio
X_test = np.array([2,8,3,16,12,144]); w_in = np.array([3,12,0.1,6, 7,0.5]); b_in = -2
f_wb = model(X_test, w_in, b_in)


####  GRADIENT DESCENT ALGORITHM

# Algoritmo per calcolare la cost function regolarizzata
def cost_function(X, y, w, b, lambda_):
    m = X.shape[0]
    n = len(w)
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 * m)
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = (lambda_/(2*m)) * reg_cost
    total_cost = cost + reg_cost
    return total_cost

# Algoritmo per calcolare il gradiente regolarizzato
def gradient(X, y, w, b, lambda_):
    m, n = X.shape
    k = len(w)
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    for j in range(k):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]
    return dj_db, dj_dw

# Funzione per il gradiente discendente
def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_db, dj_dw = gradient(X, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000: 
            J_history.append(cost_function(X, y, w, b, lambda_))
        # Stampa il costo a intervalli di 10 volte o per il numero di iterazioni se < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}")
    return w, b, J_history 


####  APPLICARE GRADIENT DESCENT 

# Facciamo i grafici per 4 valori di alpha (con lambda=0) e con 4 valori di lambda con il migliore alpha
initial_w = np.zeros(X.shape[1]); initial_b = 0.; iterations = 10000

alpha_values = [1e-15, 1e-16, 1e-17, 1e-18]
plt.figure(figsize=(14, 10))
for i, alpha in enumerate(alpha_values):
    _,_,J_history = gradient_descent(X, y, initial_w, initial_b, alpha=alpha, num_iters=iterations, lambda_=0)
    plt.subplot(2, 2, i + 1)
    plt.plot(J_history); plt.title(f'Alpha = {alpha}'); plt.xlabel('Iterations'); plt.ylabel('Cost J')
plt.tight_layout(); plt.show()

lambda_values = [0.05, 0.1, 0.5, 1]
alpha_scelto = 1e-15
plt.figure(figsize=(14, 10))
for i, lambda_ in enumerate(lambda_values):
    _,_,J_history = gradient_descent(X, y, initial_w, initial_b, alpha=alpha_scelto, num_iters=iterations, lambda_=lambda_)
    plt.subplot(2, 2, i + 1); plt.plot(J_history); plt.title(f'Lambda = {lambda_}'); plt.xlabel('Iterations'); plt.ylabel('Cost J')
plt.tight_layout(); plt.show()


# Facciamo ora il gradient descent con i migliori valori di alpha e lambda
lambda_scelto = 0.5
initial_w = np.zeros(X.shape[1]); initial_b = 0.; iterations = 1000000; alpha = alpha_scelto; lambda_ = lambda_scelto
w_final, b_final, _ = gradient_descent(X, y, initial_w, initial_b, alpha, iterations, lambda_)




























