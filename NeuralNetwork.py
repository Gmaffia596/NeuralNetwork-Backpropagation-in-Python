import numpy as np

# Training VARs
epochs = 10000
learning_rate = .05

# Network Structure - INPUT / HIDDEN / OUTPUT
input = 2
hidden = [3]
output = 2

structure = np.array([[input], hidden, [output]])

# Set weights matrices
weights = []
layers = np.concatenate(structure);
for i in range(0, len(layers) - 1):
	weights.append(np.random.rand(layers[i], layers[i + 1]))

# Set errors array
errors = []
# Set outputs array
outputs = []

# Methods
def execute_forward(input, weights_matrix_index = 0):
	global outputs
	if weights_matrix_index > 0:
		input = sigmoid(input)
	outputs.append(input)
	if weights_matrix_index == len(weights):
		return input
	else:
		sum_of_previous_layer = np.dot(input, weights[weights_matrix_index])
		return execute_forward(sum_of_previous_layer, weights_matrix_index + 1)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def train_network(input_dataset, target_dataset, epochs, learning_rate):
	global errors
	for epoc in range(0, epochs):
		for data_row in range(0, input_dataset.shape[0]):
			train_network_main(input_dataset[data_row,:], target_dataset[data_row,:], learning_rate)

def train_network_main(input, target, learning_rate):
	global errors, outputs
	output = execute_forward(input)
	backpropagate_error(np.multiply(np.multiply(target - output, output), 1 - output).T, len(weights) - 1)
	update_weights(learning_rate)
	errors = []
	outputs = []

def backpropagate_error(error, weights_matrix_index):
	global errors, outputs
	errors.insert(0, error)
	if weights_matrix_index > 0:
		transported_error = np.multiply(np.multiply(np.dot(weights[weights_matrix_index], error), outputs[weights_matrix_index].T), 1 - outputs[weights_matrix_index].T)
		backpropagate_error(transported_error, weights_matrix_index - 1)

def update_weights(learning_rate):
	global errors, outputs
	for i in range(0, len(weights)):
		weights[i] += learning_rate * (errors[i] * outputs[i]).T

# Datasets
input_dataset = np.matrix([

		[1, 0],
		[0, 1],

	])

target_dataset = np.matrix([

		[0, 1],
		[1, 0],

	])

# Train network
train_network(input_dataset, target_dataset, epochs, learning_rate)

# Test network
print(execute_forward(np.array([0, 0])))
# Expected output is [ .5 .5 ] - because it is a mid of [ 0 1 ] and [ 1 0 ]