!pip install nnfs
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# Definisco la classe dei neuroni
class Layer_Dense:
          def __init__(self, n_inputs, n_neurons):
                    self.weights = 0.10*np.random.randn(n_inputs, n_neurons)  
                    self.biases = np.zeros((1, n_neurons))
          def forward(self, inputs):
                    self.output = np.dot(inputs, self.weights) + self.biases
class Activation_ReLU:
          def forward(self, inputs):
                    self.output = np.maximum(0, inputs)
class Activation_Softmax:
          def forward(self, inputs):
                    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  ## Sottrazione per evitare overflow dovuto a exp
                    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
                    self.output = probabilities
class Loss:
          def calculate(self, output, y):
                    sample_losses = self.forward(output, y)
                    data_loss = np.mean(sample_losses)
                    return data_loss
class Loss_CategoricalCrossEntropy(Loss):
          def forward(self, y_pred, y_true):
                    samples = len(y_pred)
                    y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
                    if len(y_true) == 1:
                              correct_confidencces = y_pred_clipped[range(samples), y_true]
                    elif len(y_true.shape) == 2:
                              correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
                    negative_log_likelihoods = -np.sum(y_pred_clipped*y_true, axis=1)
                    return negative_log_likelihoods

# In questo caso utilizzo dei dati random, ma X ed y sono i miei dati che voglio analizzare
X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)  ## Imposto il numero di inputs e di neuroni 
activation1 = Activation_ReLU()  ## Imposto la funzione di attivazione
dense2 = Layer_Dense(3,3)  ## Numero di input (uguale a layer1) e di neuroni
activation2 = Activation_Softmax()  ## Imposto la seconda funzione di attivazione
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)
