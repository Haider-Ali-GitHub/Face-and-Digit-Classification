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

# Example usage
images_path = 'data/facedata/facedatatrain'
labels_path = 'data/facedata/facedatatrainlabels'
validation_images_path = 'data/facedata/facedatavalidation'
validation_labels_path = 'data/facedata/facedatavalidationlabels'

# Load training data
face_images, face_labels = main.load_data_and_labels(images_path, labels_path, 70)
flat_face_images = data_reader.flatten_images(face_images)

# Load validation data
val_face_images, val_face_labels = main.load_data_and_labels(validation_images_path, validation_labels_path, 70)
flat_val_face_images = data_reader.flatten_images(val_face_images)

# Initialize and train perceptron
perceptron = Perceptron()
training_inputs = np.array(flat_face_images, dtype=np.float32)
labels = np.array(face_labels, dtype=np.float32)
validation_inputs = np.array(flat_val_face_images, dtype=np.float32)
validation_labels = np.array(val_face_labels, dtype=np.float32)

perceptron.train(training_inputs, labels, validation_inputs, validation_labels)

# Load test data
test_images_path = 'data/facedata/facedatatest'
test_labels_path = 'data/facedata/facedatatestlabels'
test_face_images, test_face_labels = main.load_data_and_labels(test_images_path, test_labels_path, 70)
flat_test_face_images = data_reader.flatten_images(test_face_images)

# Convert test data for model evaluation
test_inputs = np.array(flat_test_face_images, dtype=np.float32)
test_labels = np.array(test_face_labels, dtype=np.float32)

# Train the model with training and validation data
perceptron.train(training_inputs, labels, validation_inputs, validation_labels)

# Evaluate the model using test data
test_accuracy = perceptron.accuracy(test_inputs, test_labels)
print(f"Final accuracy of the perceptron on the test set: {test_accuracy:.2f}%")
