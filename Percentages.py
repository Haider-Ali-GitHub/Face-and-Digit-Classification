from data_reader import load_data_and_labels, flatten_images, load_images_from_file, create_one_hot_labels_from_file, transform_to_binary_values, load_integer_labels_from_file
import face_perceptron
import face_nueral
import numpy as np
import time
import config
import digit_neural
import digit_perceptron
from digit_NaiveBayes import NaiveBayesClassifier
from faces_NaiveBayes import NaiveBayesClassifier




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

def train_and_evaluate_perceptron(percentage):
    print(f"\n PERCEPTRON: Preparing data for {percentage}% of Face training data...")
    
    # Load the full dataset
    training_inputs, labels = load_and_prepare_data(
        face_paths['train_images'], face_paths['train_labels'], config.config.get('FACE_IMAGE_SIZE'))
    validation_inputs, validation_labels = load_and_prepare_data(
        face_paths['validation_images'], face_paths['validation_labels'], config.config.get('FACE_IMAGE_SIZE'))
    test_inputs, test_labels = load_and_prepare_data(
        face_paths['test_images'], face_paths['test_labels'], config.config.get('FACE_IMAGE_SIZE'))

    # Determine the subset size based on the percentage
    subset_size = int(len(training_inputs) * (percentage / 100.0))
    training_inputs = training_inputs[:subset_size]
    labels = labels[:subset_size]

    # Initialize/train perceptron
    print(f"\nPERCEPTRON: Training with {percentage}% of data...")
    start_time = time.time()
    perceptron = face_perceptron.Perceptron()
    perceptron.train(training_inputs, labels, validation_inputs, validation_labels)
    training_time = time.time() - start_time
    print(f"PERCEPTRON: Face Training Completed (Training time: {training_time:.2f}s)")

    # Evaluate model using testing data
    test_accuracy = perceptron.accuracy(test_inputs, test_labels)
    print("PERCEPTRON: Face Validation Completed")
    print(f"PERCEPTRON: Final Face Accuracy on Face Test Set: {test_accuracy:.2f}% \n")


def train_and_evaluate_neural_network(percentage):
    print(f"NEURAL NETWORK: Preparing {percentage}% of Face data...")

    # Load data to train the neural network
    training_images, training_labels = load_data_and_labels(
        face_paths['train_images'], face_paths['train_labels'], 70)
    validation_images, validation_labels = load_data_and_labels(
        face_paths['validation_images'], face_paths['validation_labels'], 70)
    test_images, test_labels = load_data_and_labels(
        face_paths['test_images'], face_paths['test_labels'], 70)

    # Flatten images and convert labels
    training_data = np.array(flatten_images(training_images), dtype=int)
    validation_data = np.array(flatten_images(validation_images), dtype=int)
    test_data = np.array(flatten_images(test_images), dtype=int)

    training_labels = face_nueral.one_hot_encode(np.array(training_labels, dtype=int), 2)
    validation_labels = np.array(validation_labels, dtype=int)
    test_labels = np.array(test_labels, dtype=int)

    # Calculate subset size for training based on the percentage
    subset_size = int(len(training_data) * (percentage / 100.0))
    training_data = training_data[:subset_size]
    training_labels = training_labels[:subset_size]

    # Initialize/train the neural network
    print(f"NEURAL NETWORK: Training with {percentage}% of Face data...")
    start_time = time.time()
    nn = face_nueral.NeuralNetwork(training_data.shape[1], 300, 2)
    nn.train(training_data, training_labels, lr=0.001, epochs=100, batch_size=32)
    training_time = time.time() - start_time
    print(f"NEURAL NETWORK: Training Completed  (Training time: {training_time:.2f}s)")

    # Evaluate the neural network
    validation_predictions = nn.predict(validation_data)
    test_predictions = nn.predict(test_data)

    validation_accuracy = face_nueral.calculate_accuracy(validation_labels, validation_predictions)
    test_accuracy = face_nueral.calculate_accuracy(test_labels, test_predictions)
    elapsed_time = time.time() - start_time

    print("NEURAL NETWORK: Validation Completed")
    print(f"NEURAL NETWORK: Face Validation Accuracy: {100 * validation_accuracy:.2f}%")
    print(f"NEURAL NETWORK: Face Test Accuracy: {100 * test_accuracy:.2f}%  (Elapsed time: {elapsed_time:.2f}s)")


def train_and_evaluate_naive_bayes(percentage):
    print(f"\nNAIVE BAYES - Faces: Training with {percentage}% of the data")
    
    # Define paths
    face_paths = {
        'train_images': 'data/facedata/facedatatrain',
        'train_labels': 'data/facedata/facedatatrainlabels',
        'validation_images': 'data/facedata/facedatavalidation',
        'validation_labels': 'data/facedata/facedatavalidationlabels',
        'test_images': 'data/facedata/facedatatest',
        'test_labels': 'data/facedata/facedatatestlabels'
    }

    # Load and prepare training data
    train_images, train_labels = load_data_and_labels(face_paths['train_images'], face_paths['train_labels'], 70)
    train_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(train_images)]
    subset_size = int(len(train_data) * (percentage / 100.0))
    train_data = train_data[:subset_size]
    train_labels = train_labels[:subset_size]

    # Load and prepare validation and test data
    validation_images, validation_labels = load_data_and_labels(face_paths['validation_images'], face_paths['validation_labels'], 70)
    validation_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(validation_images)]
    test_images, test_labels = load_data_and_labels(face_paths['test_images'], face_paths['test_labels'], 70)
    test_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(test_images)]

    # Initialize the Naive Bayes classifier
    nb_classifier = NaiveBayesClassifier(legalLabels=[str(i) for i in range(2)])

    # Training the classifier
    start_time = time.time()
    nb_classifier.train(train_data, train_labels)
    training_time = time.time() - start_time
    print(f"Naive Bayes: Training Completed (Training time: {training_time:.2f}s)")

    # Validate the classifier
    validation_predictions = nb_classifier.classify(validation_data)
    validation_accuracy = sum(int(pred == true) for pred, true in zip(validation_predictions, validation_labels)) / len(validation_labels)
    elapsed_time = time.time() - start_time
    print(f"Face Validation Accuracy: {validation_accuracy * 100:.2f}% (Elapsed time: {elapsed_time:.2f}s)")

    # Test the classifier
    test_predictions = nb_classifier.classify(test_data)
    test_accuracy = sum(int(pred == true) for pred, true in zip(test_predictions, test_labels)) / len(test_labels)
    print(f"Face Test Accuracy: {test_accuracy * 100:.2f}% (Elapsed time: {elapsed_time:.2f}s)")

def train_and_evaluate_digit_perceptron(percentage):
    print(f"\nPERCEPTRON - Digit: Training with {percentage}% of data")
    # File paths
    test_images_path = "data/digitdata/testimages"
    test_labels_path = "data/digitdata/testlabels"
    training_images_path = "data/digitdata/trainingimages"
    training_labels_path = "data/digitdata/traininglabels"
    validation_images_path = "data/digitdata/validationimages"
    validation_labels_path = "data/digitdata/validationlabels"

    # Load/process training data
    training_data = np.array(load_images_from_file(training_images_path))
    training_labels = np.array(create_one_hot_labels_from_file(training_labels_path))
    training_data = transform_to_binary_values(training_data)
    training_data = np.reshape(training_data, (5000, 560)) 

    # Select a subset of training data based on the percentage
    subset_size = int(len(training_data) * (percentage / 100.0))
    training_data = training_data[:subset_size]
    training_labels = training_labels[:subset_size]

    # Load/process validation and testing data
    validation_data = np.array(load_images_from_file(validation_images_path))
    validation_labels = np.array(load_integer_labels_from_file(validation_labels_path))
    validation_data = transform_to_binary_values(validation_data)
    validation_data = np.reshape(validation_data, (1000, 560))

    testing_data = np.array(load_images_from_file(test_images_path))
    testing_labels = np.array(load_integer_labels_from_file(test_labels_path))
    testing_data = transform_to_binary_values(testing_data)
    testing_data = np.reshape(testing_data, (1000, 560))

    # Perceptron training
    beginning_time = time.time()
    print("Training Perceptron...")
    perceptron_weights, perceptron_biases = digit_perceptron.train_perceptron(training_data, training_labels, 100, 0.01)
    training_time = time.time() - beginning_time
    print(f"PERCEPTRON: Training Completed (Elapsed training time: {training_time:.2f}s)")

    # Validate perceptron
    perceptron_accuracy = digit_perceptron.predict(validation_data, validation_labels, perceptron_weights, perceptron_biases)
    elapsed_time = time.time() - beginning_time
    print("PERCEPTRON: Validation Completed")
    print(f"PERCEPTRON: Validation Accuracy: {perceptron_accuracy:.2f}% (Elapsed time: {elapsed_time:.2f}s)")

    # Test the perceptron
    perceptron_accuracy = digit_perceptron.predict(testing_data, testing_labels, perceptron_weights, perceptron_biases)
    elapsed_time = time.time() - beginning_time
    print("PERCEPTRON: Testing Completed")
    print(f"PERCEPTRON: Testing Accuracy: {perceptron_accuracy:.2f}% (Elapsed time: {elapsed_time:.2f}s)")


def train_and_evaluate_digit_neural_network(percentage):
    print(f"\nNEURAL NETWORK - Digit: Training with {percentage}% of data")
    # File paths
    test_images_path = "data/digitdata/testimages"
    test_labels_path = "data/digitdata/testlabels"
    training_images_path = "data/digitdata/trainingimages"
    training_labels_path = "data/digitdata/traininglabels"
    validation_images_path = "data/digitdata/validationimages"
    validation_labels_path = "data/digitdata/validationlabels"

    # Load and process training data
    training_data = np.array(load_images_from_file(training_images_path))
    training_labels = np.array(create_one_hot_labels_from_file(training_labels_path))
    training_data = transform_to_binary_values(training_data)
    training_data = np.reshape(training_data, (5000, 560)) 

    # Select a subset of training data based on the percentage
    subset_size = int(len(training_data) * (percentage / 100.0))
    training_data = training_data[:subset_size]
    training_labels = training_labels[:subset_size]

    print("Training Neural Network...")
    nn = digit_neural.NeuralNetwork(input_size=560, hidden_size=300, output_size=10)
    beginning_time = time.time()
    nn.train(training_data, training_labels, lr=0.001, epochs=100, batch_size=32)
    training_time = time.time() - beginning_time
    print(f"NEURAL NETWORK: Training Completed (Elapsed training time: {training_time:.2f}s)")

    # Load/process validation and testing data
    validation_data = np.array(load_images_from_file(validation_images_path))
    validation_labels = np.array(load_integer_labels_from_file(validation_labels_path))
    validation_data = transform_to_binary_values(validation_data)
    validation_data = np.reshape(validation_data, (1000, 560))

    testing_data = np.array(load_images_from_file(test_images_path))
    testing_labels = np.array(load_integer_labels_from_file(test_labels_path))
    testing_data = transform_to_binary_values(testing_data)
    testing_data = np.reshape(testing_data, (1000, 560))

    # Validate and test neural network
    validation_predictions = digit_neural.predict(validation_data, nn.weights_input_hidden, nn.weights_hidden_process, nn.bias_hidden, nn.bias_output)
    test_predictions = digit_neural.predict(testing_data, nn.weights_input_hidden, nn.weights_hidden_process, nn.bias_hidden, nn.bias_output)

    validation_accuracy = np.mean(np.argmax(validation_predictions, axis=1) == validation_labels)
    test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == testing_labels)
    elapsed_time = time.time() - beginning_time

    print("NEURAL NETWORK Validation Completed")
    print(f"NEURAL NETWORK Validation Accuracy: {100*validation_accuracy:.2f}% (Elapsed time: {elapsed_time:.2f}s)")
    print("NEURAL NETWORK Testing Completed")
    print(f"NEURAL NETWORK Testing Accuracy: {100*test_accuracy:.2f}% (Elapsed time: {elapsed_time:.2f}s)")



def train_and_evaluate_digit_naive_bayes(percentage):
    print(f"\nNAIVE BAYES - Digit: Training with {percentage}% of data")
    # File paths
    train_images_path = 'data/digitdata/trainingimages'
    train_labels_path = 'data/digitdata/traininglabels'
    validation_images_path = 'data/digitdata/validationimages'
    validation_labels_path = 'data/digitdata/validationlabels'
    test_images_path = 'data/digitdata/testimages'
    test_labels_path = 'data/digitdata/testlabels'
    
    # Load and process training data
    train_images, train_labels = load_data_and_labels(train_images_path, train_labels_path, 28)
    train_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(train_images)]

    # Determine the subset of data to use based on the percentage
    subset_size = int(len(train_data) * (percentage / 100.0))
    train_data = train_data[:subset_size]
    train_labels = train_labels[:subset_size]

    # Load and prepare validation and test data
    validation_images, validation_labels = load_data_and_labels(validation_images_path, validation_labels_path, 28)
    validation_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(validation_images)]
    test_images, test_labels = load_data_and_labels(test_images_path, test_labels_path, 28)
    test_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(test_images)]

    # Initialize the Naive Bayes classifier with possible digit labels
    nb_classifier = NaiveBayesClassifier(legalLabels=[str(i) for i in range(10)])  # Labels are strings from '0' to '9'

    # Training the classifier
    print("Training Naive Bayes Classifier...")
    first_time = time.time()
    nb_classifier.train(train_data, train_labels)
    training_time = time.time() - first_time
    print(f"Naive Bayes: Training Completed (Training time: {training_time:.2f}s)")

    # Validate the classifier
    print("Validating Naive Bayes Classifier...")
    validation_predictions = nb_classifier.classify(validation_data)
    validation_accuracy = sum(int(pred == true) for pred, true in zip(validation_predictions, validation_labels)) / len(validation_labels)
    elapsed_time = time.time() - first_time
    print(f"Validation Accuracy: {validation_accuracy * 100:.2f}% (Elapsed time: {elapsed_time:.2f}s)")

    # Test the classifier
    print("Testing Naive Bayes Classifier...")
    test_predictions = nb_classifier.classify(test_data)
    test_accuracy = sum(int(pred == true) for pred, true in zip(test_predictions, test_labels)) / len(test_labels)
    elapsed_time = time.time() - first_time
    print(f"Test Accuracy: {test_accuracy * 100:.2f}% (Elapsed time: {elapsed_time:.2f}s)")


def train_with_percentage(train_function, percentage):
    print(f"\nTraining with {percentage}% of the data.")
    # Adjust the function to handle this percentage
    train_function(percentage)

def main():
    percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    print("\nFACES\n------")
    for percentage in percentages:
        print(f"\nTesting with {percentage}% of the training data for FACE DATASET:")
        print("-------------------------------------------------------------")
        train_and_evaluate_perceptron(percentage)
        train_and_evaluate_neural_network(percentage)
        train_and_evaluate_naive_bayes(percentage)

    print("\nDIGITS\n----------")
    for percentage in percentages:
        print(f"\nTesting with {percentage}% of the training data for DIGIT DATASET:")
        print("---------------------------------------------------------------")
        train_and_evaluate_digit_perceptron(percentage)
        train_and_evaluate_digit_neural_network(percentage)
        train_and_evaluate_digit_naive_bayes(percentage)


if __name__ == "__main__":
    main()