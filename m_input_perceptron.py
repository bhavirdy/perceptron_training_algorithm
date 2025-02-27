import numpy as np
import training_data_generator
import argparse

class MInputPerceptron:
    def __init__(self, num_inputs):
        self.weights = np.random.uniform(-1, 1, num_inputs)
        self.bias = np.random.uniform(-1, 1)

    def classify(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        return 1 if weighted_sum + self.bias >= 0 else 0

    def train(self, training_data, labels, epochs, learning_rate):
        print(f"Initial weights: {self.weights}, Initial bias: {self.bias}")
        for epoch in range(epochs):
            
            weight_changed = False

            for inputs, label in zip(training_data, labels):
                classification = self.classify(inputs)
                # Calculate loss
                loss = label - classification
                # Update weights
                for i in range(len(self.weights)):
                    new_weight = self.weights[i] + learning_rate * loss * inputs[i]
                    if self.weights[i] != new_weight:
                        weight_changed = True
                    self.weights[i] = new_weight
                # Update bias
                self.bias += learning_rate * loss

            # If no weight was updated during this epoch, stop the training
            if not weight_changed:
                print(f"Training stopped at epoch {epoch + 1} due to no changes in weights.")
                break
        
        print(f"Final weights: {self.weights}, Final bias: {self.bias}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("num_datapoints", type=int, help="Number of datapoints N")
    parser.add_argument("num_inputs", type=int, help="Number of inputs m")

    args = parser.parse_args()

    # Linearly seperable dataset
    training_data, labels = training_data_generator.generate_training_data(num_datapoints=args.num_datapoints, num_inputs=args.num_inputs)

    m_input_perceptron = MInputPerceptron(num_inputs=args.num_inputs)

    m_input_perceptron.train(training_data, labels, epochs = 500, learning_rate = 0.1)

    correct = 0
    for inputs, label in zip(training_data, labels):
        classification = m_input_perceptron.classify(inputs)
        print(f"Inputs: {inputs}, Classified As: {classification}, Expected Classification: {label}")
        if classification == label:
            correct += 1
    accuracy = correct / len(training_data)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")