import copy, math
import numpy as np
import matplotlib.pyplot as plt

# Supponiamo di dover prevedere prevedere il prezzo di una casa (variabile dipendente) 
# Questo può essere influenzato da fattori come la dimensione, il numero di stanze, l'età della casa, la vicinanza dal centro
# Si tratta di una regressione lineare con variabili multiple
# Il modello sarà del tipo f_wb = w_1*x_1 + w_2*x_2 + ... + b
# In questo caso dobbiamo usare la regolarizzazione per evitare l'overfitting


#############################################################################################
#############################################################################################
####  DATASET

# Apriamo il dataset (supponiamo di aver già montato gdrive su colab)
datadir = datadir + 'Regression_&_Classification/'
data = np.loadtxt(datadir+'Linear_Regression_Multiple_House.csv', delimiter=',', skiprows=1)






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
def model(x, w, b): 
    p = np.dot(x, w) + b     
    return p  
  
# Per fare una previsione, selezioniamo un nuovo vettore
w = -0.05; b = 8
f_wb = model(X, w_init, b_init)




















