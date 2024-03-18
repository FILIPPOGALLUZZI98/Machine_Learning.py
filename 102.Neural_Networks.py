# Usiamo le activation functions per fare le trasformazioni non lineari, introducendo
# la non-linearità al posto della linearità
# Modifichiamo la funzione lineare y = wx+b --> y = activation(wx+b)


## ACTIVATION FUNCTION
# La sigmoid function mappa gli input in valori compresi tra 0 e 1
def sigmoid(x):
  return 1/(1+np.exp(-x))
# la ReLU restituisce gli input se sono positivi, mentre restituisce 0 se sono negativi
def relu(x):
  return np.maximum(0,x)
# La leaky ReLU consente un piccolo output diverso da zero per valori input negativi,
# impedendo la completa soppressione delle informazioni
def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha*x, x)
# La tanh mappa l'input in valori compresi tra -1 e 1
def tanh(x):
    return np.tanh(x)
# La softmax converte un vettore di valori reali in una distribuzione di probabilità 
# su classi multiple
def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores)


## MULTIPLE LAYERS
# Una rete neurale con due layers può essere rappresentata dall'equazione 
# y = w2*relu(w1*x + b1) + b2

# Una lambda function calcola una previsione lineare basata sull'input x, usando
# valori predefiniti dei pesi e dei bias
prediction = lambda x, w1=.2, b=1.99: x * w1 + b
# Applichiamo la ReLU alle previsioni lineari
layer1_1 = np.maximum(0, prediction(x))
# Ed aggiungiamo un altro layer
layer1_2 = np.maximum(0, prediction(x, .3, -2))
# Abbiamo introdotto delle nonlinearità. Aggiungiamo un altro layer
layer1_3 = np.maximum(0, prediction(x, .6, -2))
plt.plot(x, layer1_1+layer1_2+layer1_3)



## LOSS
# Possiamo usare l'errore quadratico medio per calcolare l'errore output/tartget
def calculate_mse(target, predicted):
    return (target - predicted) ** 2
target = np.array([[9], [13], [5], [-2], [-1]])
# Usando il gradient descent è essenziale determinare il gradiente della loss function, chre
# rappresenta il rate of change. Indica come cambia la loss function al cambiare dell'input
def gradient_mse(actual, predicted):
    return predicted - actual

## BACKPROPAGATION
# Inverte il processo forward per distribuire il gradiente nei differenti parametri della rete
# Calcoliamo la derivata parziale della loss function per ogni parametro
# Il gradiente dell'output viene usato come perso nellayer2. We perform this update by multiplying
# the input to layer 2 and the output from layer 1 in the forward pass by the L2 output gradient
# The bias is updated by taking the average of the output gradient
# Ora propaghiamo il gradiente allayer1 moltiplicando l'output per i pesi del layer2
# Passiamo poi questo gradiente attraverso la ReLU function e lo utilizziamo per aggiornare
# i pesi ed i bias del layer1
output_gradient = gradient_mse(actual, output)
with exp():
    l2_w_gradient =  l1_activated.T @ output_gradient
l2_w_gradient
l2_w_gradient =  l1_activated.T @ output_gradient
# Calcoliamo la derivata per il bias
with exp():
    l2_b_gradient =  np.mean(output_gradient, axis=0)
l2_b_gradient
# Per aggiornare i pesi ed i bias del layer2 sottraiamo il gradiente dai valori di
# w e b, scaled by the learning rate (questo ci aiuta per non ottenere aggiornamenti
# che siano troppo grandi)
lr = 1e-4  ## Set the learning rate
with exp():
    # Update the bias values
    l2_bias = l2_bias - l2_b_gradient * lr
    # Update the weight values
    l2_weights = l2_weights - l2_w_gradient * lr
l2_weights
# Poi proseguiamo nel calcolare i gradidenti per il layer1
# Gli output del layer1 sono ottenuti scalando gli inputs con i corrispondenti pesi, 
# risultando come gli output del layer2
with exp():
    # Calculate the gradient on the output of layer 1
    l1_activated_gradient = output_gradient @ l2_weights.T
l1_activated_gradient
with exp():
    l1_output_gradient = l1_activated_gradient * np.heaviside(l1_output, 0)
l1_output_gradient
# Adesso calcoliamo il gradiente del layer1
l1_w_gradient =  input.T @ l1_output_gradient
l1_b_gradient = np.mean(l1_output_gradient, axis=0)
l1_weights -= l1_w_gradient * lr
l1_bias -= l1_b_gradient * lr


## STEPS
# 1) Fare il forward pass attraverso la rete ed ottenere l'output
# 2) Calcolare il gradiente per i network outputs usando la funzione mse_grad
# 3) Per ogni layer della rete:
#      i. determinare il gradienteper la pre-nonlinearity output
#     ii. calcolare il gradiente per i pesi
#    iii. calcolare il gradiente per i bias
#     iv. determinare il gradiente per gli inputs al layer
# 4) Aggiornare i parametri della rete usando il gradient descend




## CREARE UNA CLASSE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statistics import mean
from typing import Dict, List, Tuple

np.random.seed(42)

class Neural:
    
    def __init__(self, layers: List[int], epochs: int, 
                 learning_rate: float = 0.001, batch_size: int=32,
                 validation_split: float = 0.2, verbose: int=1):
        self._layer_structure: List[int] = layers
        self._batch_size: int = batch_size
        self._epochs: int = epochs
        self._learning_rate: float = learning_rate
        self._validation_split: float = validation_split
        self._verbose: int = verbose
        self._losses: Dict[str, float] = {"train": [], "validation": []}
        self._is_fit: bool = False
        self.__layers = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # validation split
        X, X_val, y, y_val = train_test_split(X, y, test_size=self._validation_split, random_state=42)
        # initialization of layers
        self.__layers = self.__init_layers()
        for epoch in range(self._epochs):
            epoch_losses = []
            for i in range(1, len(self.__layers)):
                # forward pass
                x_batch = X[i:(i+self._batch_size)]
                y_batch = y[i:(i+self._batch_size)]
                pred, hidden = self.__forward(x_batch)
                # calculate loss
                loss = self.__calculate_loss(y_batch, pred)
                epoch_losses.append(np.mean(loss ** 2))
                #backward
                self.__backward(hidden, loss)
            valid_preds, _ = self.__forward(X_val)
            train_loss = mean(epoch_losses)
            valid_loss = np.mean(self.__calculate_mse(valid_preds,y_val))
            self._losses["train"].append(train_loss)
            self._losses["validation"].append(valid_loss)
            if self._verbose:
                print(f"Epoch: {epoch} Train MSE: {train_loss} Valid MSE: {valid_loss}")
        self._is_fit = True
        return
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._is_fit == False:
            raise Exception("Model has not been trained yet.")
        pred, hidden = self.__forward(X)
        return pred
    
    def plot_learning(self) -> None:
        plt.plot(self._losses["train"],label="loss")
        plt.plot(self._losses["validation"],label="validation")
        plt.legend()
    
    def __init_layers(self) -> List[np.ndarray]:
        layers = []
        for i in range(1, len(self._layer_structure)):
            layers.append([
                np.random.rand(self._layer_structure[i-1], self._layer_structure[i]) / 5 - .1,
                np.ones((1,self._layer_structure[i]))
            ])
        return layers
    
    def __forward(self, batch: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        hidden = [batch.copy()]
        for i in range(len(self.__layers)):
            batch = np.matmul(batch, self.__layers[i][0]) + self.__layers[i][1]
            if i < len(self.__layers) - 1:
                batch = np.maximum(batch, 0)
            # Store the forward pass hidden values for use in backprop
            hidden.append(batch.copy())
        return batch, hidden
    
    def __calculate_loss(self,actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        "mse"
        return predicted - actual
    
    
    def __calculate_mse(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return (actual - predicted) ** 2
    
    def __backward(self, hidden: List[np.ndarray], grad: np.ndarray) -> None:
        for i in range(len(self.__layers)-1, -1, -1):
            if i != len(self.__layers) - 1:
                grad = np.multiply(grad, np.heaviside(hidden[i+1], 0))
    
            w_grad = hidden[i].T @ grad
            b_grad = np.mean(grad, axis=0)
    
            self.__layers[i][0] -= w_grad * self._learning_rate
            self.__layers[i][1] -= b_grad * self._learning_rate
            
            grad = grad @ self.__layers[i][0].T
        return


## GENARARE DUMMY DATA PER TEST + CLIENT CODE
def generate_data():
    # Define correlation values
    corr_a = 0.8
    corr_b = 0.4
    corr_c = -0.2
    
    # Generate independent features
    a = np.random.normal(0, 1, size=100000)
    b = np.random.normal(0, 1, size=100000)
    c = np.random.normal(0, 1, size=100000)
    d = np.random.randint(0, 4, size=100000)
    e = np.random.binomial(1, 0.5, size=100000)
    
    # Generate target feature based on independent features
    target = 50 + corr_a*a + corr_b*b + corr_c*c + d*10 + 20*e + np.random.normal(0, 10, size=100000)
    
    # Create DataFrame with all features
    df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'target': target})
    return df

df = generate_data()

# Separate the features and target
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.to_numpy().reshape(-1,1)
y_test = y_test.to_numpy().reshape(-1,1)

layer_structure = [X_train.shape[1],10,10,1]
nn = Neural(layer_structure, 20, 1e-5, 64, 0.2, 1)

nn.fit(X_train, y_train)

y_pred = nn.predict(X_test)
nn.plot_learning()

print("Test error: ",mean_squared_error(y_test, y_pred))






















