import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        sum = np.dot(inputs, self.weights) + self.bias
        return self.activation(sum)

    def train(self, training_inputs, labels, validation_inputs, validation_labels):
        self.weights = np.zeros(len(training_inputs[0]))
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights += self.lr * (label - prediction) * np.array(inputs)
                self.bias += self.lr * (label - prediction)
            # Validation accuracy at the end of each epoch
            val_accuracy = self.accuracy(validation_inputs, validation_labels)
            print(f"Validation Accuracy after epoch {_+1}: {val_accuracy:.2f}%")

    def accuracy(self, inputs, labels):
        predictions = [self.predict(input) for input in inputs]
        correct_predictions = sum([pred == label for pred, label in zip(predictions, labels)])
        return correct_predictions / len(labels) * 100  # Return accuracy as a percentage

