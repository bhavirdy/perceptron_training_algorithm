import random
import numpy as np
import matplotlib.pyplot as plt

class TwoWayPerceptron:
    def __init__(self):
        self.weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.bias = random.uniform(-1, 1)
        
        # History for plotting
        self.original_weights = self.weights.copy()
        self.original_bias = self.bias

    def classify(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return 1 if weighted_sum >= 0 else 0

    def train(self, training_data, labels, epochs, learning_rate):
        for _ in range(epochs):
            for inputs, label in zip(training_data, labels):
                classification = self.classify(inputs)
                # Calculate loss
                loss = label - classification
                # Update weights
                for i in range(len(self.weights)):
                    self.weights[i] += learning_rate * loss * inputs[i]
                # Update bias
                self.bias += learning_rate * loss
        self.visualise_classification(training_data, labels)

    def visualise_classification(self, training_data, labels):
        fig, axis = plt.subplots()

        # Plot training data
        for inputs, label in zip(training_data, labels):
            color = 'blue' if label == 1 else 'red'
            axis.scatter(inputs[0], inputs[1], color=color)

        # Plot classification line before training
        x1 = np.linspace(-5, 5, 100)
        x2 = (-(x1 * self.original_weights[0]) - self.original_bias) / (self.original_weights[1])
        axis.plot(x1, x2, 'r--', label='Classification Line Before Training')

        # Plot classification line after training
        x2 = (-(x1 * self.weights[0]) - self.bias) / (self.weights[1])
        axis.plot(x1, x2, 'g--', label='Classification Line After Training')

        axis.set_title("Classification Line Before and After Training")
        axis.set_xlim(-5, 5)
        axis.set_ylim(-5, 5)
        axis.spines['left'].set_position('zero')
        axis.spines['bottom'].set_position('zero')
        axis.spines['right'].set_color('none')
        axis.spines['top'].set_color('none')
        axis.set_xticks(np.arange(-5, 6, 1))
        axis.set_yticks(np.arange(-5, 6, 1))
        axis.legend()
        axis.grid(True)
        plt.show()

if __name__ == "__main__":

    two_way_perceptron = TwoWayPerceptron()

    # Linearly seperable dataset
    training_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [1, 1, 1, 0]

    two_way_perceptron.train(training_data, labels, epochs = 50, learning_rate = 0.1)

    # Test classification
    for inputs in training_data:
        print(f"Input: {inputs}, Predicted: {two_way_perceptron.classify(inputs)}")