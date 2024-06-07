import numpy as np
import matplotlib.pyplot as plt
# Importare le funzioni descritte in precedenza per singole variabili di:
# cost_function, gradient, gradient_descent 

# Supponiamo di voler usare come modello f = 1+x^2
x = np.arange(1, 16)
y = 1 + x**2

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












