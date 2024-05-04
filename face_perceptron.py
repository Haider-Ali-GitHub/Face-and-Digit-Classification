from main import flat_digit_images, digit_labels, flat_face_images, face_labels
import numpy as np
import main
import data_reader

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

    def train(self, training_inputs, labels):
        self.weights = np.zeros(len(training_inputs[0]))
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights += self.lr * (label - prediction) * np.array(inputs)
                self.bias += self.lr * (label - prediction)

    def accuracy(self, inputs, labels):
        predictions = [self.predict(input) for input in inputs]
        correct_predictions = sum([pred == label for pred, label in zip(predictions, labels)])
        return correct_predictions / len(labels) * 100  # Return accuracy as a percentage

# Example usage
images_path = 'data/facedata/facedatatrain'
labels_path = 'data/facedata/facedatatrainlabels'
face_images, face_labels = main.load_data_and_labels(images_path, labels_path, 70)
flat_face_images = data_reader.flatten_images(face_images)

perceptron = Perceptron()
training_inputs = np.array(flat_face_images, dtype=np.float32) 
labels = np.array(face_labels, dtype=np.float32)

split_index = int(len(training_inputs) * 0.963)
train_inputs, test_inputs = training_inputs[:split_index], training_inputs[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]

perceptron.train(train_inputs, train_labels)

test_accuracy = perceptron.accuracy(test_inputs, test_labels)
print(f"Accuracy of the perceptron on the test set: {test_accuracy:.2f}%")
