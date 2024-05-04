import config
import os
from data_reader import load_data_and_labels, flatten_images
import face_perceptron
import numpy as np

train_images_path = 'data/facedata/facedatatrain'
train_labels_path = 'data/facedata/facedatatrainlabels'

validation_images_path = 'data/facedata/facedatavalidation'
validation_labels_path = 'data/facedata/facedatavalidationlabels'

test_images_path = 'data/facedata/facedatatest'
test_labels_path = 'data/facedata/facedatatestlabels'

# Load training data
face_images, face_labels = load_data_and_labels(train_images_path, train_labels_path, 70)
flat_face_images = flatten_images(face_images)

# Load validation data
val_face_images, val_face_labels = load_data_and_labels(validation_images_path, validation_labels_path, 70)
flat_val_face_images = flatten_images(val_face_images)

# Load test data
test_face_images, test_face_labels = load_data_and_labels(test_images_path, test_labels_path, 70)
flat_test_face_images = flatten_images(test_face_images)

# Initialize and train perceptron
perceptron = face_perceptron.Perceptron()
training_inputs = np.array(flat_face_images, dtype=np.float32)
labels = np.array(face_labels, dtype=np.float32)
validation_inputs = np.array(flat_val_face_images, dtype=np.float32)
validation_labels = np.array(val_face_labels, dtype=np.float32)

# Train the model with training and validation data
perceptron.train(training_inputs, labels, validation_inputs, validation_labels)

# Convert test data for model evaluation
test_inputs = np.array(flat_test_face_images, dtype=np.float32)
test_labels = np.array(test_face_labels, dtype=np.float32)

# Evaluate the model using test data
test_accuracy = perceptron.accuracy(test_inputs, test_labels)
print(f"Final accuracy of the perceptron on the test set: {test_accuracy:.2f}%")
