# Implementation of sigmoid function:
def sigmoid(z):
    g = 1/(1+np.exp(-z))   
    return g
