import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100, input_dim=None, regularization=0.01, early_stop_rounds=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.input_dim = input_dim
        self.regularization = regularization
        self.early_stop_rounds = early_stop_rounds
        self.weights = None
        self.bias = 0

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        return self.activation(np.dot(inputs, self.weights) + self.bias)

    def initialize_weights(self, input_dim):
        self.weights = np.random.uniform(-0.01, 0.01, input_dim)
        self.bias = np.random.uniform(-0.01, 0.01)

    def train(self, training_inputs, labels, validation_inputs, validation_labels):
        if self.input_dim is None:
            self.input_dim = len(training_inputs[0])
        self.initialize_weights(self.input_dim)

        best_val_accuracy = 0
        no_improvement_epochs = 0

        for epoch in range(self.epochs):
            # training phase
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction

                # regularization
                self.weights = (1 - self.lr * self.regularization) * self.weights + self.lr * error * np.array(inputs)
                self.bias += self.lr * error

            # validate
            val_accuracy = self.accuracy(validation_inputs, validation_labels)
            print(f"Epoch {epoch+1} Validation Accuracy: {val_accuracy:.2f}%")

            # stop early if not changing
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            if no_improvement_epochs >= self.early_stop_rounds:
                print(f"Stopped early @ Epoch {epoch+1}. Best validation accuracy: {best_val_accuracy:.2f}%\n")
                break

    def accuracy(self, inputs, labels):
        predictions = [self.predict(input) for input in inputs]
        correct_predictions = sum([pred == label for pred, label in zip(predictions, labels)])
        return correct_predictions / len(labels) * 100
