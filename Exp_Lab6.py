import numpy as np

# Sigmoid Activation Function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Function to initialize parameters based on user input
def initialize_parameters():
    # Get user inputs for parameters
    X = np.array([float(input("Enter input x1: ")), float(input("Enter input x2: "))])  # Inputs [x1, x2]
    y_true = float(input("Enter the expected output (y): "))  # Expected output
    
    learning_rate = float(input("Enter the learning rate: "))
    
    # Weights from input to hidden layer
    w1 = float(input("Enter weight w1 (from x1 to z1): "))
    w2 = float(input("Enter weight w2 (from x2 to z1): "))
    w3 = float(input("Enter weight w3 (from bias to z1): "))
    
    w4 = float(input("Enter weight w4 (from x1 to z2): "))
    w5 = float(input("Enter weight w5 (from x2 to z2): "))
    w6 = float(input("Enter weight w6 (from bias to z2): "))
    
    # Weights from hidden layer to output layer
    w7 = float(input("Enter weight w7 (from z1 to y): "))
    w8 = float(input("Enter weight w8 (from z2 to y): "))
    w9 = float(input("Enter weight w9 (from bias to y): "))
    
    # Biases
    b1 = float(input("Enter bias b1 for z1: "))
    b2 = float(input("Enter bias b2 for z2: "))
    b3 = float(input("Enter bias b3 for y: "))
    
    return X, y_true, learning_rate, w1, w2, w3, w4, w5, w6, w7, w8, w9, b1, b2, b3

# Main function
def main():
    # Initialize parameters
    X, y_true, learning_rate, w1, w2, w3, w4, w5, w6, w7, w8, w9, b1, b2, b3 = initialize_parameters()

    # Forward propagation
    z1_input = w1 * X[0] + w2 * X[1] + w3 * b1  # Input to z1
    z1 = sigmoid(z1_input)

    z2_input = w4 * X[0] + w5 * X[1] + w6 * b2  # Input to z2
    z2 = sigmoid(z2_input)

    y_input = w7 * z1 + w8 * z2 + w9 * b3  # Input to y
    y_pred = sigmoid(y_input)

    # Compute the error (Mean Squared Error)
    error = y_true - y_pred
    print(f"Initial output: {y_pred}, Error: {error}")

    # Backpropagation
    delta_y = error * sigmoid_derivative(y_pred)

    # Gradients for weights from hidden layer to output layer
    grad_w7 = delta_y * z1
    grad_w8 = delta_y * z2
    grad_w9 = delta_y * b3

    # Calculate the gradient of the hidden layer
    delta_z1 = delta_y * w7 * sigmoid_derivative(z1)
    delta_z2 = delta_y * w8 * sigmoid_derivative(z2)

    # Gradients for weights from input layer to hidden layer
    grad_w1 = delta_z1 * X[0]
    grad_w2 = delta_z1 * X[1]
    grad_w3 = delta_z1 * b1

    grad_w4 = delta_z2 * X[0]
    grad_w5 = delta_z2 * X[1]
    grad_w6 = delta_z2 * b2

    # Update weights with the gradients
    w1 += learning_rate * grad_w1
    w2 += learning_rate * grad_w2
    w3 += learning_rate * grad_w3
    w4 += learning_rate * grad_w4
    w5 += learning_rate * grad_w5
    w6 += learning_rate * grad_w6
    w7 += learning_rate * grad_w7
    w8 += learning_rate * grad_w8
    w9 += learning_rate * grad_w9

    # Output updated weights after one backpropagation iteration
    print("\nUpdated Weights:")
    print(f"w1: {w1}, w2: {w2}, w3: {w3}")
    print(f"w4: {w4}, w5: {w5}, w6: {w6}")
    print(f"w7: {w7}, w8: {w8}, w9: {w9}")

# Run the main function
if __name__ == "__main__":
    main()

