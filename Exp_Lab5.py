import numpy as np

# Define the step function (activation function)
def step_function(x):
    return 1 if x >= 0 else 0

# Define a function to simulate a single-layer neural network with convergence tracking
def train_logic_gate(inputs, expected_output, learning_rate=0.1, epochs=10000):
    # Initialize weights and bias
    weights = np.random.rand(2)  # Two inputs
    bias = np.random.rand(1)     # One bias
    
    # Training loop
    for epoch in range(epochs):
        total_error = 0  # To track if the model has no errors
        for i in range(len(inputs)):
            # Calculate the weighted sum (input * weights + bias)
            weighted_sum = np.dot(inputs[i], weights) + bias
            # Apply the activation function (step function)
            output = step_function(weighted_sum)
            # Calculate the error (difference between expected and actual output)
            error = expected_output[i] - output
            total_error += abs(error)
            
            # Update weights and bias if there's an error
            weights += learning_rate * error * inputs[i]
            bias += learning_rate * error
        
        # If there is no error after this epoch, then stop training
        if total_error == 0:
            print(f"Converged at epoch {epoch+1}")
            break  # Stop training as the model has converged
    
    return weights, bias

# Define the dataset for AND and OR gates
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Expected output for AND gate
and_output = np.array([0, 0, 0, 1])
# Expected output for OR gate
or_output = np.array([0, 1, 1, 1])

# Train for AND gate
print("Training for AND gate:")
and_weights, and_bias = train_logic_gate(inputs, and_output)
print(f"AND Gate Weights: {and_weights}, Bias: {and_bias}")

# Train for OR gate
print("\nTraining for OR gate:")
or_weights, or_bias = train_logic_gate(inputs, or_output)
print(f"OR Gate Weights: {or_weights}, Bias: {or_bias}")

# Define a function to make predictions
def predict(inputs, weights, bias):
    predictions = []
    for i in range(len(inputs)):
        weighted_sum = np.dot(inputs[i], weights) + bias
        output = step_function(weighted_sum)
        predictions.append(output)
    return predictions

# Test the model with the trained weights and biases for both gates
print("\nTesting AND Gate:")
and_predictions = predict(inputs, and_weights, and_bias)
print(f"Predictions: {and_predictions}")

print("\nTesting OR Gate:")
or_predictions = predict(inputs, or_weights, or_bias)
print(f"Predictions: {or_predictions}")

# User prediction function
def user_prediction(weights, bias):
    print("\nEnter two binary inputs (either 0 or 1):")
    try:
        input1 = int(input("Input 1: "))
        input2 = int(input("Input 2: "))
        if input1 not in [0, 1] or input2 not in [0, 1]:
            print("Invalid input. Please enter 0 or 1.")
            return
        input_data = np.array([input1, input2])
        prediction = predict([input_data], weights, bias)
        print(f"The predicted output is: {prediction[0]}")
    except ValueError:
        print("Invalid input. Please enter integers 0 or 1.")

# Ask user for a prediction for AND Gate
print("\nPredicting for AND gate:")
user_prediction(and_weights, and_bias)

# Ask user for a prediction for OR Gate
print("\nPredicting for OR gate:")
user_prediction(or_weights, or_bias)

