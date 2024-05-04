import config
from data_reader import load_data_and_labels, flatten_images
import face_perceptron
import numpy as np

face_paths = {
    'train_images': 'data/facedata/facedatatrain',
    'train_labels': 'data/facedata/facedatatrainlabels',
    'validation_images': 'data/facedata/facedatavalidation',
    'validation_labels': 'data/facedata/facedatavalidationlabels',
    'test_images': 'data/facedata/facedatatest',
    'test_labels': 'data/facedata/facedatatestlabels'
}

digit_paths = {
    'train_images': 'data/digitdata/trainingimages',
    'train_labels': 'data/digitdata/facedatatrainlabels',
    'validation_images': 'data/facedata/facedatavalidation',
    'validation_labels': 'data/facedata/facedatavalidationlabels',
    'test_images': 'data/facedata/facedatatest',
    'test_labels': 'data/facedata/facedatatestlabels'
}

def load_and_prepare_data(image_path, label_path, image_size):
    """Load and flatten images and labels."""
    images, labels = load_data_and_labels(image_path, label_path, image_size)
    flat_images = flatten_images(images)
    return np.array(flat_images, dtype=np.float32), np.array(labels, dtype=np.float32)

def main():
    # Load and prepare data
    training_inputs, labels = load_and_prepare_data(
        face_paths['train_images'], face_paths['train_labels'], config.config.get('FACE_IMAGE_SIZE'))
    validation_inputs, validation_labels = load_and_prepare_data(
        face_paths['validation_images'], face_paths['validation_labels'], config.config.get('FACE_IMAGE_SIZE'))
    test_inputs, test_labels = load_and_prepare_data(
        face_paths['test_images'], face_paths['test_labels'], config.config.get('FACE_IMAGE_SIZE'))

    # Initialize and train perceptron
    perceptron = face_perceptron.Perceptron()
    perceptron.train(training_inputs, labels, validation_inputs, validation_labels)

    # Evaluate the model using test data
    test_accuracy = perceptron.accuracy(test_inputs, test_labels)
    print(f"Final accuracy of the perceptron on the test set: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
