import copy, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Supponiamo di dover prevedere prevedere il prezzo di una casa (variabile dipendente) 
# Questo può essere influenzato da fattori come la dimensione, il numero di stanze, l'età della casa, la vicinanza dal centro
# Si tratta di una regressione lineare con variabili multiple
# Il modello sarà del tipo f_wb = w1*x1 + w2*x2 + ... + b


#############################################################################################
#############################################################################################
####  DATASET

# Apriamo il dataset (supponiamo di aver già montato gdrive su colab)
datadir = datadir + 'Regression_&_Classification/'
file_path = datadir +'Linear_Regression_Multiple_House.csv'
df = pd.read_csv(file_path)
X = df[['Dimensione (m^2)', 'Numero di stanze', 'Età della casa (anni)', 'Distanza dal centro (km)']].values
y = df['Prezzo (€)'].values

# Plottiamo i 4 grafici
features = ['Dimensione (m^2)', 'Numero di stanze', 'Età della casa (anni)', 'Distanza dal centro (km)']
titles = ['Dimensione vs Prezzo', 'Numero di stanze vs Prezzo', 'Età della casa vs Prezzo', 'Distanza dal centro vs Prezzo']
plt.figure(figsize=(12, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i + 1)
    plt.scatter(df[feature], df['Prezzo (€)'], s = 1, color = 'k')
    plt.title(titles[i]); plt.xlabel(feature); plt.ylabel('Prezzo (€)')
plt.tight_layout(); plt.show()


#############################################################################################
#############################################################################################
####  MODELLO

# La funzione che ci fornisce la previsione (cioè il modello)
def model(x, w, b): 
    p = np.dot(x, w) + b     
    return p  
  
# Per fare una previsione, selezioniamo un nuovo vettore esempio
x_vec = np.array([[120, 5, 20, 40]])
b_init = 785; w_init = np.array([ 0.4, 18.7, -53.36, -26.42])
f_wb = model(x_vec, w_init, b_init)


#############################################################################################
#############################################################################################
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


#############################################################################################
#############################################################################################
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

# Facciamo i 4 grafici possibili









f_wb = model_single(x, w_final, b_final)
plt.plot(x, f_wb, c='b',label='Our Prediction')
plt.scatter(x, y, marker='x', c='r',label='Actual Values')
plt.show()













