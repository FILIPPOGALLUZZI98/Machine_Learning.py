# Machine learning is a technique in which you train the system to solve a problem instead 
# of explicitly programming the rules
# The goal of supervised learning tasks is to make predictions for new, unseen data
# To do that, you assume that this unseen data follows a probability distribution similar
# to the distribution of the training dataset. If in the future this distribution changes, 
# then you need to train your model again using the new training dataset.
# Another name for input data is feature, and feature engineering is the process of extracting 
# features from raw data
# Deep learning is a technique in which you let the neural network figure out by itself which features
# are important instead of applying feature engineering techniques

# One cool thing about neural network layers is that the same computations can extract information from 
# any kind of data. This means that it doesn’t matter if you’re using image data or text data. The process
# to extract meaningful information and train the deep learning model is the same for both scenarios

# By modeling the relationship between the variables as linear, you can express the dependent variable as a
# weighted sum of the independent variables. So, each independent variable will be multiplied by a vector called
# weight. Besides the weights and the independent variables, you also add another vector: the bias. It sets the 
# result when all the other independent variables are equal to zero




input_vector = np.array([1.72, 1.23])
weights_1 = np.array([1.26, 0])
weights_2 = np.array([2.17, 0.32])
# Fare il dot product del vettore input con il primo weight
dot_product_1 = np.dot(input_vector, weights_1)
# Fare il dot product del vettore input con il secondo weight
dot_product_2 = np.dot(input_vector, weights_2)

# Adesso faremoun tarin di un modello per fare previsioni che possono avere
# solo due output --> classification problem
# The target is the variable you want to predict

# If you add more layers but keep using only linear operations, then adding more
# layers would have no effect because each layer will always have some correlation
# with the input of the previous layer. This implies that, for a network with multiple
# layers, there would always be a network with fewer layers that predicts the same results
# What you want is to find an operation that makes the middle layers sometimes correlate 
# with an input and sometimes not correlate
# You can achieve this behavior by using nonlinear functions. These nonlinear functions are 
# called activation functions

# The sigmoid function is a good choice if your problem follows the Bernoulli distribution

input_vector = np.array([1.66, 1.56])
weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])
def sigmoid(x):
  return 1/(1+np.exp(-x))
def make_prediction(input_vector, weights, bias):
   layer_1 = np.dot(input_vector, weights) + bias
   layer_2 = sigmoid(layer_1)
   return layer_2
prediction = make_prediction(input_vector, weights_1, bias)




## TRAIN NEURAL NETWORKS
# In the process of training the neural network, you first assess the error and
# then adjust the weights accordingly. To adjust the weights, you’ll use the
# gradient descent and backpropagation algorithms. Gradient descent is applied to
# find the direction and the rate to update the parameters

# To understand the magnitude of the error, you need to choose a way to measure it
# The function used to measure the error is called the cost function, or loss function
# Qui useremo la mean squared error come cost function

# The goal is to change the weights and bias variables so you can reduce the error
# To understand how this works, you’ll change only the weights variable and leave the 
# bias fixed for now. You can also get rid of the sigmoid function and use only the result
# of layer_1. All that’s left is to figure out how you can modify the weights so that the error goes down
# You compute the MSE by doing 'error = np.square(prediction - target)'

# Gradient descent is the name of the algorithm used to find the direction and the rate to update the network parameters

# When it comes to your neural network, the derivative will tell you the direction you should take to update the weights variable
# If it’s a positive number, then you predicted too high, and you need to decrease the weights. If it’s a negative number, then you
# predicted too low, and you need to increase the weights

target = 0
error = np.square(prediction - target)
derivative = 2 * (prediction - target)
weights_1 = weights_1 - derivative
prediction = make_prediction(input_vector, weights_1, bias)
error = (prediction - target) ** 2

# To define a fraction for updating the weights, you use the alpha parameter, also called the learning rate. If you decrease the learning rate,
# then the increments are smaller. If you increase it, then the steps are higher. How do you know what’s the best learning rate value? By making
# a guess and experimenting with it

# The network you’re building has two layers, and since each layer has its own functions, you’re dealing with a function composition
# Since now you have this function composition, to take the derivative of the error concerning the parameters, you’ll need to use the chain rule
# In your neural network, you need to update both the weights and the bias vectors
# Applying the chain rule, the value of derror_dweights will be
derror_dweights = (
    derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
)


## BACKPROPAGATION
# Since you are starting from the end and going backward, you first need to take the partial derivative of the error with respect to the prediction
# The prediction is the result of the sigmoid function. You can take the derivative of the sigmoid function by multiplying sigmoid(x) and 1 - sigmoid(x)
# Now you’ll take the derivative of layer_1 with respect to the bias
def sigmoid_deriv(x):
     return sigmoid(x) * (1-sigmoid(x))
derror_dprediction = 2 * (prediction - target)
layer_1 = np.dot(input_vector, weights_1) + bias
dprediction_dlayer1 = sigmoid_deriv(layer_1)
dlayer1_dbias = 1
derror_dbias = (
     derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
)
# To update the weights, you follow the same process, going backward and taking the partial derivatives until you get to the weights variable
# Since you’ve already computed some of the partial derivatives, you’ll just need to compute dlayer1_dweights




## CLASS
# When instantiating a NeuralNetwork object, you need to pass the learning_rate parameter. You’ll use predict() to make a prediction
# The methods _compute_derivatives() and _update_parameters() have the computations you learned in this section
class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors

# In short, you pick a random instance from the dataset, compute the gradients, and update the weights
# and the bias. You also compute the cumulative error every 100 iterations and save those results in an array
# You’ll plot this array to visualize how the error changes during the training process





# Make a prediction
learning_rate = 0.1
neural_network = NeuralNetwork(learning_rate)
neural_network.predict(input_vector)
array([0.79412963])


input_vectors = np.array([
[3, 1.5],
[2, 1],
[4, 1.5],
[3, 4],
[3.5, 0.5],
[2, 0.5],
[5.5, 1],
[1, 1],
])

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
learning_rate = 0.1
neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 10000)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")


















