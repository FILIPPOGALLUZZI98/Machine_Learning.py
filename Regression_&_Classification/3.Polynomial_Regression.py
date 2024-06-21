import copy, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Il modello sarà del tipo f_wb = w1*x1 + w2*x1^2 + w3*x2^(-1/2) + w4*x3^3 + ... + b
# In questo caso dobbiamo usare la regolarizzazione per evitare l'overfitting

# Supponiamo di dover prevedere la resa di un campo in base ad alcune variabili come
# la quantità di fertilizzanti usati, di acqua e ore di sole
# Usiamo solo queste 3 variabili (anche se nel dataset ce ne sono di più)


#############################################################################################
#############################################################################################
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


#############################################################################################
#############################################################################################
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


#############################################################################################
#############################################################################################
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


#############################################################################################
#############################################################################################
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
initial_w = np.zeros(X.shape[1]); initial_b = 0.; iterations = 1000000; alpha = 1.0e-5; lambda_ = 0
w_final, b_final, _ = gradient_descent(X, y, initial_w, initial_b, alpha, iterations, lambda_)


# Facciamo i 2 grafici delle possibili variabili per vedere come influiscono sulla resa
plt.figure(figsize=(10, 10))
for i, feature in enumerate(features):
    plt.subplot(1, 2, i + 1)
    plt.scatter(df[feature], df['Prezzo (€)'], s=1, color='k', label='Dati')
    x_vals = np.linspace(df[feature].min(), df[feature].max(), 100).reshape(-1, 1)
    X_vals = np.full((x_vals.shape[0], X.shape[1]), df[features].mean(axis=0))
    X_vals[:, i] = x_vals.flatten()
    y_vals = model(X_vals, w_final, b_final)
    plt.plot(x_vals, y_vals, color='r', label='Regressione'); plt.title(titles[i])
    plt.xlabel(feature); plt.ylabel('Prezzo (€)'); plt.legend()
plt.tight_layout(); plt.show()

# Per fare una previsione della resa 
x_vec = np.array([[120, 5, 20, 40]])
f_wb = model(x_vec, w_final, b_final); f_wb





























