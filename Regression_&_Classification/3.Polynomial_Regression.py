import numpy as np
import matplotlib.pyplot as plt
# Importare le funzioni descritte in precedenza per singole variabili di:
# cost_function, gradient, gradient_descent 

# Codice per singola variable, ma può essere adattato anche a più variabili guardando il
# file '2.Linear_Regression_Multiple.py'

# Supponiamo di voler usare come modello f = 1+x^2
x = np.arange(1, 16)
y = [4.9,7.8,14.7,27.6,28.5,41.4,53.3,69.2,90.1,120. ,135.9, 168.8,180.7,210.6,250.5]
# Facciamo il GD algorithm cercando una relazione lineare
w_in = 0; b_in = 0; alpha = 0.001; N = 100000
model_w,model_b = gradient_descent_single(x,y,w_in,b_in, alpha, N ,cost_function_single, gradient_single)
# Mostriamo il grafico
plt.scatter(x, y, marker='x', c='r'); plt.plot(x, np.dot(x,model_w) + model_b); plt.show()
# Questo non è un buon fit perché stiamo modellizzando qualcosa di quadratico con un modello lineare 

# Creiamo allora una Engineered feature
X = x**2
w_in = 0; b_in = 0; alpha = 0.00001; N = 100000
model_w,model_b = gradient_descent_single(X,y,w_in,b_in, alpha, N ,cost_function_single, gradient_single)
plt.plot(x, np.dot(X,model_w) + model_b); plt.scatter(x, y, marker='x', c='r'); plt.show()












