from data_reader import load_data_and_labels, flatten_images, load_images_from_file, create_one_hot_labels_from_file, transform_to_binary_values, load_integer_labels_from_file
import face_perceptron
import face_nueral
import numpy as np
import config
import digit_neural
import digit_perceptron

# THIS FILE IS A LITTLE WEIRD. All face-data relating things are located below in functions and variables.
# ALL DIGIT RELATED THINGS ARE DIRECTLY IN THE MAIN FUNCTION BELOW

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
    perceptron = face_perceptron.Perceptron()
    perceptron.train(training_inputs, labels, validation_inputs, validation_labels)
    print("PERCEPTRON: Training Completed")

    # evaluate model using testing data
    test_accuracy = perceptron.accuracy(test_inputs, test_labels)
    print("PERCEPTRON: Validation Completed")
    print(f"PERCEPTRON: Final Accuracy on Test Set: {test_accuracy:.2f}%")

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
    nn = face_nueral.NeuralNetwork(training_data.shape[1], 300, 2)
    nn.train(training_data, training_labels, lr=0.001, epochs=100, batch_size=32)
    print("NEURAL NETWORK: Training Completed")

    # evaluate neural network
    validation_predictions = nn.predict(validation_data)
    test_predictions = nn.predict(test_data)

    validation_accuracy = face_nueral.calculate_accuracy(validation_labels, validation_predictions)
    test_accuracy = face_nueral.calculate_accuracy(test_labels, test_predictions)

    print("NEURAL NETWORK: Validation Completed")
    print(f"NEURAL NETWORK Validation Accuracy: {100*validation_accuracy:.2f}%")
    print(f"NEURAL NETWORK Test Accuracy: {100*test_accuracy:.2f}%")

def main():
    print("\nFACES\n------")
    print("\nPERCEPTRON- Faces")
    print("--------------------")
    train_and_evaluate_perceptron()
    print("\nNEURAL NETWORK- Faces")
    print("----------------------")
    train_and_evaluate_neural_network()


    print("\nDIGITS\n----------")

    # file paths
    test_images_path = "data/digitdata/testimages"
    test_labels_path = "data/digitdata/testlabels"
    training_images_path = "data/digitdata/trainingimages"
    training_labels_path = "data/digitdata/traininglabels"
    validation_images_path = "data/digitdata/validationimages"
    validation_labels_path = "data/digitdata/validationlabels"

    # load/process training data
    training_data = np.array(load_images_from_file(training_images_path))
    training_labels = np.array(create_one_hot_labels_from_file(training_labels_path))
    training_data = transform_to_binary_values(training_data)
    training_data = np.reshape(training_data, (5000, 560))  # for neural network!!

    print("\nNEURAL NETWORK- Digit")
    print("--------------------")

    # initialize and train the neural network
    print("Training Neural Network...")
    nn = digit_neural.NeuralNetwork(input_size=560, hidden_size=300, output_size=10)
    nn.train(training_data, training_labels, lr=0.001, epochs=100, batch_size=32)
    print("\nNEURAL NETWORK: Training Completed")

    # load/process validation data
    validation_data = np.array(load_images_from_file(validation_images_path))
    validation_labels = np.array(load_integer_labels_from_file(validation_labels_path))
    validation_data = transform_to_binary_values(validation_data)
    validation_data = np.reshape(validation_data, (1000, 560))

    testing_data = np.array(load_images_from_file(test_images_path))
    testing_labels = np.array(load_integer_labels_from_file(test_labels_path))
    testing_data = transform_to_binary_values(testing_data)
    testing_data = np.reshape(testing_data, (1000, 560))

    # validate neural network
    nn_predicted_labels = digit_neural.predict(validation_data, nn.weights_input_hidden, nn.weights_hidden_process, nn.bias_hidden, nn.bias_output)
    nn_accuracy = np.mean(np.argmax(nn_predicted_labels, axis=1) == validation_labels)
    print("NEURAL NETWORK Validation Completed")
    print(f"NEURAL NETWORK Validation Accuracy: {100*nn_accuracy:.2f}%\n")

    # testing the neural Network
    nn_predicted_labels = digit_neural.predict(testing_data, nn.weights_input_hidden, nn.weights_hidden_process, nn.bias_hidden, nn.bias_output)
    nn_accuracy = np.mean(np.argmax(nn_predicted_labels, axis=1) == testing_labels)
    print("NEURAL NETWORK Testing Completed")
    print(f"NEURAL NETWORK Testing Accuracy: {100*nn_accuracy:.2f}%\n")

    


    print("\nPERCEPTRON- Digit")
    print("--------------------")

    # reuse training data for perceptron
    training_data_perceptron = transform_to_binary_values(np.array(load_images_from_file(training_images_path)))

    # perceptron training 
    print("Training Perceptron...")
    perceptron_weights, perceptron_biases = digit_perceptron.train_perceptron(training_data_perceptron, training_labels, 100, 0.01)
    print("\nPERCEPTRON: Training Completed")

    # load validation data for perceptron
    validation_data_perceptron = transform_to_binary_values(np.array(load_images_from_file(validation_images_path)))
    validation_data_perceptron = np.reshape(validation_data_perceptron, (len(validation_data_perceptron), 560))

    # validate perceptron
    perceptron_accuracy = digit_perceptron.predict(validation_data_perceptron, validation_labels, perceptron_weights, perceptron_biases)
    print("PERCEPTRON: Validation Completed")
    print(f"PERCEPTRON: Validation Accuracy: {perceptron_accuracy:.2f}%\n")

    # load testing data for perceptron
    test_data_perceptron = transform_to_binary_values(np.array(load_images_from_file(test_images_path)))
    test_data_perceptron = np.reshape(test_data_perceptron, (len(test_data_perceptron), 560))

    # test the perceptron
    perceptron_accuracy = digit_perceptron.predict(test_data_perceptron, testing_labels, perceptron_weights, perceptron_biases)
    print("PERCEPTRON: Testing Completed")
    print(f"PERCEPTRON: Testing Accuracy: {perceptron_accuracy:.2f}%\n")

if __name__ == "__main__":
    main()



























if __name__ == "__main__":
    main()
