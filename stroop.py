import numpy as np
import random

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))



weights_input_hidden_color = np.random.rand(2,3)
weights_input_hidden_word = np.random.rand(2,3)

bias_hidden_color = np.zeros((2,1))
bias_hidden_word = np.zeros((2,1))

weights_hidden_output_color = np.random.rand(2,2)
weights_hidden_output_word = np.random.rand(2,2)

bias_output = np.zeros((2,1))



# Backpropagation
def backward_propagation(X_color, X_word, target):
    final_output, hidden_color_output, hidden_word_output = feed_forward(X_color, X_word)
    
    # Compute output layer error
    output_error = final_output - target
    
    # Compute hidden to output layer gradients
    gradient_hidden_output_color = np.dot(output_error, hidden_color_output.T)
    gradient_hidden_output_word = np.dot(output_error, hidden_word_output.T)

    # Compute output layer bias gradient
    gradient_bias_output = output_error

    # Compute hidden layer errors
    hidden_color_input = np.dot(weights_input_hidden_color, X_color) + bias_hidden_color
    hidden_word_input = np.dot(weights_input_hidden_word, X_word) + bias_hidden_word
    
    hidden_output_error_color = np.dot(weights_hidden_output_color.T, output_error) * sigmoid_derivative(hidden_color_input)
    hidden_output_error_word = np.dot(weights_hidden_output_word.T, output_error) * sigmoid_derivative(hidden_word_input)

    # Compute input to hidden layer gradients
    gradient_input_hidden_color = np.dot(hidden_output_error_color, X_color.T)
    gradient_input_hidden_word = np.dot(hidden_output_error_word, X_word.T)

    # Compute hidden layer bias gradients
    gradient_bias_hidden_color = hidden_output_error_color
    gradient_bias_hidden_word = hidden_output_error_word

    return gradient_input_hidden_color, gradient_input_hidden_word, gradient_hidden_output_color, gradient_hidden_output_word, gradient_bias_hidden_color, gradient_bias_hidden_word, gradient_bias_output

# Update weights and biases using gradients
def update_parameters(learning_rate, gradient_input_hidden_color, gradient_input_hidden_word, gradient_hidden_output_color, gradient_hidden_output_word, gradient_bias_hidden_color, gradient_bias_hidden_word, gradient_bias_output):
    # Update input to hidden layer weights
    global weights_input_hidden_color
    global weights_input_hidden_word
    weights_input_hidden_color -= learning_rate * gradient_input_hidden_color
    weights_input_hidden_word -= learning_rate * gradient_input_hidden_word

    # Update hidden to output layer weights
    global weights_hidden_output_color
    global weights_hidden_output_word
    weights_hidden_output_color -= learning_rate * gradient_hidden_output_color
    weights_hidden_output_word -= learning_rate * gradient_hidden_output_word

    # Update hidden layer biases
    global bias_hidden_color
    global bias_hidden_word
    bias_hidden_color -= learning_rate * gradient_bias_hidden_color
    bias_hidden_word -= learning_rate * gradient_bias_hidden_word

    # Update output layer bias
    global bias_output
    bias_output -= learning_rate * gradient_bias_output

    return

# Training phase 
iterations = 0 
while(iterations < 1000):
    learning_rate = 0.1
    random_number = random.random()
    if(random_number < 0.6):
        X_color = np.array([[1],[0],[1]]) 
        X_word = np.array([[0],[0],[0]])   
        target = np.array([[1],[0]])

        # Perform backpropagation
        gradient_input_hidden_color, gradient_input_hidden_word, gradient_hidden_output_color, gradient_hidden_output_word, gradient_bias_hidden_color, gradient_bias_hidden_word, gradient_bias_output = backward_propagation(X_color, X_word, target)

        # Update parameters
        update_parameters(learning_rate, gradient_input_hidden_color, gradient_input_hidden_word, gradient_hidden_output_color, gradient_hidden_output_word, gradient_bias_hidden_color, gradient_bias_hidden_word, gradient_bias_output)

        X_color = np.array([[0],[1],[1]]) 
        X_word = np.array([[0],[0],[0]])   
        target = np.array([[0],[1]])   

        # Perform backpropagation
        gradient_input_hidden_color, gradient_input_hidden_word, gradient_hidden_output_color, gradient_hidden_output_word, gradient_bias_hidden_color, gradient_bias_hidden_word, gradient_bias_output = backward_propagation(X_color, X_word, target)

        # Update parameters
        update_parameters(learning_rate, gradient_input_hidden_color, gradient_input_hidden_word, gradient_hidden_output_color, gradient_hidden_output_word, gradient_bias_hidden_color, gradient_bias_hidden_word, gradient_bias_output)

    
    X_color = np.array([[0],[0],[0]]) 
    X_word = np.array([[1],[1],[0]])   
    target = np.array([[1],[0]]) 

    # Perform backpropagation
    gradient_input_hidden_color, gradient_input_hidden_word, gradient_hidden_output_color, gradient_hidden_output_word, gradient_bias_hidden_color, gradient_bias_hidden_word, gradient_bias_output = backward_propagation(X_color, X_word, target)

    # Update parameters
    update_parameters(learning_rate, gradient_input_hidden_color, gradient_input_hidden_word, gradient_hidden_output_color, gradient_hidden_output_word, gradient_bias_hidden_color, gradient_bias_hidden_word, gradient_bias_output)

    X_color = np.array([[0],[0],[0]]) 
    X_word = np.array([[1],[0],[1]])   
    target = np.array([[0],[1]]) 

    # Perform backpropagation
    gradient_input_hidden_color, gradient_input_hidden_word, gradient_hidden_output_color, gradient_hidden_output_word, gradient_bias_hidden_color, gradient_bias_hidden_word, gradient_bias_output = backward_propagation(X_color, X_word, target)

    # Update parameters
    update_parameters(learning_rate, gradient_input_hidden_color, gradient_input_hidden_word, gradient_hidden_output_color, gradient_hidden_output_word, gradient_bias_hidden_color, gradient_bias_hidden_word, gradient_bias_output)

    iterations += 1

# Test Stimuli 

# Color naming

# Control 
X_color = [[1],[0],[1]]
X_word = [[0],[0],[0]]

output = feed_forward(X_color, X_word)[0] 
confidence = output[0] / output[1]
print(confidence)
