import numpy as np
import matplotlib.pyplot as plt

# Supponiamo di voler usare come modello f = 1+x^2
x = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y = 1 + x**2

# Facciamo il GD algorithm 
w_in = 0; b_in = 0; alpha = 0.001; N = 100000
model_w,model_b,J,p = gradient_descent_single(x,y,w_in,b_in, alpha, N ,cost_function_single, gradient_single)
# Mostriamo il grafico
plt.scatter(x, y, marker='x', c='r'); plt.plot(x, np.dot(x,model_w) + model_b); plt.show()
# Questo non è un buon fit perché stiamo modellizzando qualcosa di quadratico con un modello lineare 


# Creiamo allora una Engineered feature
X = x**2
model_w,model_b,J,p = gradient_descent_single(X,y,w_in,b_in, alpha, N ,cost_function_single, gradient_single)
plt.scatter(x, y, marker='x', c='r'); plt.plot(x, np.dot(X,model_w) + model_b); plt.show()












