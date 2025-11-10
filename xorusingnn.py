import numpy as np

# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and biases
np.random.seed(1)
W1 = np.random.randn(2, 2)  # input → hidden
b1 = np.random.randn(1, 2)
W2 = np.random.randn(2, 1)  # hidden → output
b2 = np.random.randn(1, 1)

lr = 0.1  # learning rate

# Training (backpropagation)
for epoch in range(10000):

    # Forward Propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, W2) + b2
    predicted_output = sigmoid(output_input)

    # Loss calculation (MSE)
    error = y - predicted_output

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    hidden_layer_error = d_predicted_output.dot(W2.T)
    d_hidden_output = hidden_layer_error * sigmoid_derivative(hidden_output)

    # Update weights
    W2 += hidden_output.T.dot(d_predicted_output) * lr
    b2 += np.sum(d_predicted_output, axis=0, keepdims=True) * lr

    W1 += X.T.dot(d_hidden_output) * lr
    b1 += np.sum(d_hidden_output, axis=0, keepdims=True) * lr

# Final prediction
print("\nFinal predictions after training:")
print(predicted_output.round())
