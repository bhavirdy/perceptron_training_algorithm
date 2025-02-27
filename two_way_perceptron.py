import random
import matplotlib.pyplot as plt
import numpy as np

class TwoWayPerceptron:

    def __init__(self, training_vectors, training_targets):
        self.training_vectors = training_vectors
        self.training_targets = training_targets
        self.weights = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        self.original_weights = self.weights.copy()
        self.theta = random.uniform(-0.5, 0.5)
        self.original_theta = self.theta
        self.learning_rate = 0.1
        self.loss_history = []

    def get_output(self, training_vector):
        net = (self.weights[0] * training_vector[0]) + (self.weights[1] * training_vector[1])
        if net >= self.theta:
            return 1
        return 0

    def update_weights(self, training_vector, y, t):
        self.weights[0] = self.weights[0] + (self.learning_rate * (t - y) * training_vector[0])
        self.weights[1] = self.weights[1] + (self.learning_rate * (t - y) * training_vector[1])

    def update_theta(self, y, t):
        self.theta = self.theta - (self.learning_rate * (t - y))
    
    def train(self):
        stopping_condition = False
        epochs = 0

        while stopping_condition == False:
            total_loss = 0
            for i, training_vector in enumerate(training_vectors):
                y = self.get_output(training_vector)
                t = training_targets[i]
                total_loss += abs(t - y)
                self.update_weights(training_vector, y, t)
                self.update_theta(y, t)  

            self.loss_history.append(total_loss)

            if (total_loss == 0):
                stopping_condition = True
                print("Successfully trained in " + str(epochs) + " epochs.")
                self.plot_figure(epochs)
            
            epochs += 1
    
    def plot_figure(self, epochs):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Subplot 1: Training Data and Decision Boundary
        ax1 = axes[0]
        for i, training_vector in enumerate(self.training_vectors):
            color = 'blue' if self.training_targets[i] == 1 else 'red'
            ax1.scatter(training_vector[0], training_vector[1], color=color)

        x = np.linspace(-5, 5, 100)
        y_initial = (self.original_theta - (self.original_weights[0] * x)) / self.original_weights[1]
        ax1.plot(x, y_initial, 'y--', label='Boundary Before Training')

        y_final = (self.theta - (self.weights[0] * x)) / self.weights[1]
        ax1.plot(x, y_final, 'g--', label='Decision Boundary')

        ax1.set_title("Training Data and Decision Boundary")
        ax1.legend()
        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 5)
        ax1.grid(True)

        # Subplot 2: Total Loss vs. Epochs
        ax2 = axes[1]
        ax2.plot(range(epochs + 1), self.loss_history, marker='o', linestyle='-')
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Total Loss")
        ax2.set_title("Total Loss vs. Number of Epochs")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    training_vectors = [[0, 0], [0, 1], [1, 0], [1, 1]]
    training_targets = [1, 1, 1, 0]

    two_way_perceptron = TwoWayPerceptron(training_vectors, training_targets)

    two_way_perceptron.train()