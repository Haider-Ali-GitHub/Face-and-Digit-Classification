import config 
from data_reader import load_data_and_labels, flatten_images
import face_perceptron
import numpy as np

face_train_images_path = 'data/facedata/facedatatrain'
face_train_labels_path = 'data/facedata/facedatatrainlabels'

face_validation_images_path = 'data/facedata/facedatavalidation'
face_validation_labels_path = 'data/facedata/facedatavalidationlabels'

face_test_images_path = 'data/facedata/facedatatest'
face_test_labels_path = 'data/facedata/facedatatestlabels'

# Load training data
face_images, face_labels = load_data_and_labels(face_train_images_path, face_train_labels_path, config.config.get('FACE_IMAGE_SIZE'))
flat_face_images = flatten_images(face_images)

# Load validation data
val_face_images, val_face_labels = load_data_and_labels(face_validation_images_path, face_validation_labels_path, config.config.get('FACE_IMAGE_SIZE'))
flat_val_face_images = flatten_images(val_face_images)

# Load test data
test_face_images, test_face_labels = load_data_and_labels(face_test_images_path, face_test_labels_path, config.config.get('FACE_IMAGE_SIZE'))
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
