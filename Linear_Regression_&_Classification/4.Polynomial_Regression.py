import numpy as np
import matplotlib.pyplot as plt

# Supponiamo di voler usare come modello f = 1+x^2

# Create target data
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1)

# Facciamo il GD algorithm
model_w,model_b = gradient_descent(X,y,iterations=1000, alpha = 1e-2)
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
plt.plot(x,X@model_w + model_b, label="Predicted Value");  plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()
# Questo non è un buon fit perché stiamo modellizzando qualcosa di quadratico con un modello lineare 

# Creiamo una Engineered feature
X = x**2
X = X.reshape(-1, 1)  #X should be a 2-D Matrix
model_w,model_b = gradient_descent(X, y, iterations=10000, alpha = 1e-5)
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


# Possiamo anche aggiungere altre engineered features
X = np.c_[x, x**2, x**3]
model_w,model_b = gradient_descent(X, y, iterations=10000, alpha=1e-7)
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()












