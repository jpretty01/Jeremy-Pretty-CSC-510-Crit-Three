# Jeremy Pretty
# CSC 510 Week 3
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train(inputs, outputs, epochs, learning_rate):
    input_layer_size = inputs.shape[1]
    hidden_layer_size = 2
    output_layer_size = 1

    # Initialize weights and biases
    hidden_weights = np.random.uniform(size=(input_layer_size, hidden_layer_size))
    hidden_biases = np.random.uniform(size=(1, hidden_layer_size))
    output_weights = np.random.uniform(size=(hidden_layer_size, output_layer_size))
    output_biases = np.random.uniform(size=(1, output_layer_size))

    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(inputs, hidden_weights) + hidden_biases
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, output_weights) + output_biases
        predicted_output = sigmoid(output_layer_input)

        # Calculate the error
        error = outputs - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        # Backpropagation
        error_hidden_layer = d_predicted_output.dot(output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Update weights and biases
        output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        output_biases += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        hidden_weights += inputs.T.dot(d_hidden_layer) * learning_rate
        hidden_biases += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    return hidden_weights, hidden_biases, output_weights, output_biases

def predict(inputs, hidden_weights, hidden_biases, output_weights, output_biases):
    hidden_layer_input = np.dot(inputs, hidden_weights) + hidden_biases
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_biases
    predicted_output = sigmoid(output_layer_input)
    return predicted_output

# Define the input and output data for XOR function
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = np.array([[0], [1], [1], [0]])

# Train the ANN
epochs = 10000
learning_rate = 0.1
hidden_weights, hidden_biases, output_weights, output_biases = train(input_data, output_data, epochs, learning_rate)

# Test the ANN
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = predict(test_data, hidden_weights, hidden_biases, output_weights, output_biases)
print("Predicted output for XOR function:")
print(predicted_output)
