import numpy as np
import os
from data_reader import read_images, read_labels
import main
def perceptron_train(X, y, epochs=1, learning_rate=0.01):
    """
    Trains a perceptron model.

    Args:
    X (np.array): The input features, each row representing an image (flattened).
    y (np.array): The target labels corresponding to each image.
    epochs (int): Number of times to iterate over the training dataset.
    learning_rate (float): The step size at each iteration.

    Returns:
    np.array: The weights after training.
    """
    # Initialize weights to zero
    weights = np.zeros(X.shape[1] + 1)  # +1 for the bias term

    # Training loop
    for _ in range(epochs):
        for i in range(len(X)):
            # Calculate the dot product + bias
            activation = np.dot(X[i], weights[1:]) + weights[0]
            # Apply the step function
            prediction = 1 if activation >= 0 else 0
            # Update weights and bias
            weights[1:] += learning_rate * (y[i] - prediction) * X[i]
            weights[0] += learning_rate * (y[i] - prediction)
    return weights

def perceptron_predict(X, weights):
    """
    Make predictions with a perceptron model.

    Args:
    X (np.array): The input features, each row representing an image (flattened).
    weights (np.array): The trained weights.

    Returns:
    np.array: Predictions for each input.
    """
    # Calculate the dot product + bias for each input
    activations = np.dot(X, weights[1:]) + weights[0]
    # Apply the step function
    return np.where(activations >= 0, 1, 0)

FACE_DIR = 'data/facedata'
DIGIT_DIR = 'data/digitdata'

FACE_IMAGE_SIZE = 70
DIGIT_IMAGE_SIZE = 28

face_datasets = {
    'test': ('facedatatest', 'facedatatestlabels'),
    'train': ('facedatatrain', 'facedatatrainlabels'),
    'validation': ('facedatavalidation', 'facedatavalidationlabels')
}

digit_datasets = {
    'test': ('testimages', 'testlabels'),
    'train': ('trainingimages', 'traininglabels'),
    'validation': ('validationimages', 'validationlabels')
}

for name, (img_file, label_file) in digit_datasets.items():
    images_path = os.path.join(DIGIT_DIR, img_file)
    labels_path = os.path.join(DIGIT_DIR, label_file)
    digit_image_labels = main.load_data_and_labels(images_path, labels_path, DIGIT_IMAGE_SIZE)

for name, (img_file, label_file) in face_datasets.items():
    images_path = os.path.join(FACE_DIR, img_file)
    labels_path = os.path.join(FACE_DIR, label_file)
    digit_image_labels = main.load_data_and_labels(images_path, labels_path, FACE_IMAGE_SIZE)

# Example of training and predicting
# Flatten the images for perceptron input
flattened_images = [np.array(image).flatten() for image in images]  # Assuming 'images' is available from the loading function
flattened_images = np.array(flattened_images)
binary_labels = np.array(labels)  # Assuming binary labels and 'labels' is available

# Convert labels from string or other types to integers if needed
# binary_labels = np.array([int(label) for label in labels])

# Train the perceptron
weights = perceptron_train(flattened_images, binary_labels, epochs=10, learning_rate=0.01)

# Predict using the trained model
predictions = perceptron_predict(flattened_images, weights)
