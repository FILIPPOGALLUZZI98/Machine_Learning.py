# Questi sono input unici dati ad ogni nuerone (supponiamo di averne 3). Per ora sono arbitrari
# Gli input in un neurone potrebbero arrivare sia da valori esterni che da un altro neurone precedente
# Stiamo considerando 3 input in un solo neuronen, quindi avremo un solo bias
inputs = [1, 2, 3, 2.5]  
weights = [0.2, 0.8, -0.5, 1]
bias = 2
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] +  inputs[3]*weights[3] + bias
print(output)


# Passiamo ora a 3 neuroni con 4 input (e quindi 3 output)
inputs = [1, 2, 3, 2.5]
# Ci saranno 3 set di pesi ed ognuno di essi avrà 4 valori più 3 bias
weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2
bias2 = 3
bias3 = 0.5
output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] +  inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] +  inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] +  inputs[3]*weights3[3] + bias3]
print(output)

# Versione migliorata:
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
          neuron_output = 0
          for n_input, weight in zip(inputs, neuron_weights):
                    neuron_output += n_input*weight
          neuron_output += neuron_bias
          layer_outputs.append(neuron_output)
print(layer_outputs)


# Dot product
import numpy as np
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
output = np.dot(weights, inputs) + bias  ## weights devono essere primi (indicizzazione della matrice)
print(output)


# BATCH
import numpy as np
inputs = [[1, 2, 3, 2.5], 
           [2, 5, -1, 2], 
           [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
# Dobbiamo usare la trasposta in questo caso, altrimenti non torna
output = np.dot(inputs, np.array(weights).T) + biases  ## In questo caso abbiamo invertito inputs e weights

# Possiamo anche usare due layer in modo che il primo sia input del secondo
inputs = [[1, 2, 3, 2.5], 
           [2, 5, -1, 2], 
           [-1.5, 2.7, 3.3, -0.8]]
weights1 = [[0.2, 0.8, -0.5, 1], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
weights2 = [[0.1, -0.14, 0.5], 
           [-0.5, 0.12, -0.33], 
           [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)


# Invece di scrivere molti layer possiamo usare una classe
X = [[1, 2, 3, 2.5], 
     [2, 5, -1, 2], 
     [-1.5, 2.7, 3.3, -0.8]]
# Definiamo gli hidden layer
















