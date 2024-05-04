import config
from data_reader import load_data_and_labels, flatten_images
import face_perceptron
import face_nueral
import numpy as np
import time

# file paths
face_paths = {
    'train_images': 'data/facedata/facedatatrain',
    'train_labels': 'data/facedata/facedatatrainlabels',
    'validation_images': 'data/facedata/facedatavalidation',
    'validation_labels': 'data/facedata/facedatavalidationlabels',
    'test_images': 'data/facedata/facedatatest',
    'test_labels': 'data/facedata/facedatatestlabels'
}

def load_and_prepare_data(image_path, label_path, image_size):

    images, labels = load_data_and_labels(image_path, label_path, image_size)
    flat_images = flatten_images(images)
    return np.array(flat_images, dtype=np.float32), np.array(labels, dtype=np.float32)

def train_and_evaluate_perceptron():
    
    print("PERCEPTRON: Preparing data...")
    training_inputs, labels = load_and_prepare_data(
        face_paths['train_images'], face_paths['train_labels'], config.config.get('FACE_IMAGE_SIZE'))
    validation_inputs, validation_labels = load_and_prepare_data(
        face_paths['validation_images'], face_paths['validation_labels'], config.config.get('FACE_IMAGE_SIZE'))
    test_inputs, test_labels = load_and_prepare_data(
        face_paths['test_images'], face_paths['test_labels'], config.config.get('FACE_IMAGE_SIZE'))

    # initailize/train perceptron
    print("\nPERCEPTRON: Training...")
    start_time = time.time()
    perceptron = face_perceptron.Perceptron()
    perceptron.train(training_inputs, labels, validation_inputs, validation_labels)
    training_time = time.time() - start_time
    print(f"PERCEPTRON: Training Completed (Training time: {training_time:.2f})")

    # evaluate model using testing data
    test_accuracy = perceptron.accuracy(test_inputs, test_labels)
    print("PERCEPTRON: Validation Completed")
    print(f"PERCEPTRON: Final Accuracy on Test Set: {test_accuracy:.2f}% ")

def train_and_evaluate_neural_network():
    
    print("NEURAL NETWORK: Preparing data...")
    # load data to train neural network 
    training_images, training_labels = load_data_and_labels(
        face_paths['train_images'], face_paths['train_labels'], 70)
    validation_images, validation_labels = load_data_and_labels(
        face_paths['validation_images'], face_paths['validation_labels'], 70)
    test_images, test_labels = load_data_and_labels(
        face_paths['test_images'], face_paths['test_labels'], 70)

    # flatten images and convert labels
    training_data = np.array(flatten_images(training_images), dtype=int)
    validation_data = np.array(flatten_images(validation_images), dtype=int)
    test_data = np.array(flatten_images(test_images), dtype=int)

    training_labels = face_nueral.one_hot_encode(np.array(training_labels, dtype=int), 2)
    validation_labels = np.array(validation_labels, dtype=int)
    test_labels = np.array(test_labels, dtype=int)

    # nitialize/train neural network
    print("NEURAL NETWORK: Training...")
    start_time = time.time()
    nn = face_nueral.NeuralNetwork(training_data.shape[1], 300, 2)
    nn.train(training_data, training_labels, lr=0.001, epochs=100, batch_size=32)
    training_time = time.time()-start_time
    print(f"NEURAL NETWORK: Training Completed  (Training time: {training_time:.2f})")

    # evaluate neural network
    validation_predictions = nn.predict(validation_data)
    test_predictions = nn.predict(test_data)

    validation_accuracy = face_nueral.calculate_accuracy(validation_labels, validation_predictions)
    test_accuracy = face_nueral.calculate_accuracy(test_labels, test_predictions)
    elapsed_time = time.time() - start_time

    print("NEURAL NETWORK: Validation Completed")
    print(f"NEURAL NETWORK Validation Accuracy: {100*validation_accuracy:.2f}%")
    print(f"NEURAL NETWORK Test Accuracy: {100*test_accuracy:.2f}%  (Elapsed time: {elapsed_time:.2f})")

def main():
    train_and_evaluate_perceptron()
    train_and_evaluate_neural_network()

if __name__ == "__main__":
    main()
