import random
import numpy as np

# Define linear function
def linear_function(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

def generate_training_data(num_datapoints, num_inputs):
    weights  = np.random.uniform(-1, 1, num_inputs)
    bias  = np.random.uniform(-1, 1)

    training_data = []
    labels = []

    for _ in range(num_datapoints):
        inputs = np.random.uniform(-5, 5, num_inputs)

        result = linear_function(inputs, weights, bias)

        label = 1 if result >= 0 else 0

        training_data.append(inputs)
        labels.append(label)

    return training_data, labels