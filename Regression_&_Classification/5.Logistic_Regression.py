# Implementation of sigmoid function:
def sigmoid(z):
    g = 1/(1+np.exp(-z))   
    return g

# Cost Function for Logistic Regression
def cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost / m
    return cost




