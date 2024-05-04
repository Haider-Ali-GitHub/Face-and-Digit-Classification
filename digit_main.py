import numpy as np
from data_reader import load_images_from_file, create_one_hot_labels_from_file, transform_to_binary_values, load_integer_labels_from_file
import digit_neural
import digit_perceptron

def main():
    # Set up data paths
    test_images_path = "data/digitdata/testimages"
    test_labels_path = "data/digitdata/testlabels"
    training_images_path = "data/digitdata/trainingimages"
    training_labels_path = "data/digitdata/traininglabels"
    validation_images_path = "data/digitdata/validationimages"
    validation_labels_path = "data/digitdata/validationlabels"

    # Load and preprocess training data
    training_data = np.array(load_images_from_file(training_images_path))
    training_labels = np.array(create_one_hot_labels_from_file(training_labels_path))
    training_data = transform_to_binary_values(training_data)
    training_data = np.reshape(training_data, (5000, 560))  # Specific to neural network

    # Initialize and train the Neural Network
    print("\nNEURAL NETWORK: Training...")
    nn = digit_neural.NeuralNetwork(input_size=560, hidden_size=300, output_size=10)
    nn.train(training_data, training_labels, lr=0.001, epochs=100, batch_size=32)
    print("\nNEURAL NETWORK: Training Completed")

    # Load and preprocess validation data
    validation_data = np.array(load_images_from_file(validation_images_path))
    validation_labels = np.array(load_integer_labels_from_file(validation_labels_path))
    validation_data = transform_to_binary_values(validation_data)
    validation_data = np.reshape(validation_data, (1000, 560))

    testing_data = np.array(load_images_from_file(test_images_path))
    testing_labels = np.array(load_integer_labels_from_file(test_labels_path))
    testing_data = transform_to_binary_values(testing_data)
    testing_data = np.reshape(testing_data, (1000, 560))

    # Validate the Neural Network
    nn_predicted_labels = digit_neural.predict(validation_data, nn.weights_input_hidden, nn.weights_hidden_process, nn.bias_hidden, nn.bias_output)
    nn_accuracy = np.mean(np.argmax(nn_predicted_labels, axis=1) == validation_labels)
    print("NEURAL NETWORK Validation Completed")
    print(f"NEURAL NETWORK Validation Accuracy: {100*nn_accuracy:.2f}%\n")

    # Validate the Neural Network
    nn_predicted_labels = digit_neural.predict(testing_data, nn.weights_input_hidden, nn.weights_hidden_process, nn.bias_hidden, nn.bias_output)
    nn_accuracy = np.mean(np.argmax(nn_predicted_labels, axis=1) == testing_labels)
    print("NEURAL NETWORK Testing Completed")
    print(f"NEURAL NETWORK Testing Accuracy: {100*nn_accuracy:.2f}%\n")





    # Reuse the same training data but for perceptron; no reshape needed
    training_data_perceptron = transform_to_binary_values(np.array(load_images_from_file(training_images_path)))

    # Execute perceptron training process
    print("\nPERCEPTRON: Training...")
    perceptron_weights, perceptron_biases = digit_perceptron.train_perceptron(training_data_perceptron, training_labels, 100, 0.01)
    print("\nPERCEPTRON: Training Completed")

    # Load validation data for perceptron and transform
    validation_data_perceptron = transform_to_binary_values(np.array(load_images_from_file(validation_images_path)))
    validation_data_perceptron = np.reshape(validation_data_perceptron, (len(validation_data_perceptron), 560))

    # Validate the perceptron model
    perceptron_accuracy = digit_perceptron.predict(validation_data_perceptron, validation_labels, perceptron_weights, perceptron_biases)
    print("PERCEPTRON: Validation Completed")
    print(f"PERCEPTRON: Validation Accuracy: {perceptron_accuracy:.2f}%\n")

    # Load testing data for perceptron and transform
    test_data_perceptron = transform_to_binary_values(np.array(load_images_from_file(test_images_path)))
    test_data_perceptron = np.reshape(test_data_perceptron, (len(test_data_perceptron), 560))

    # Validate the perceptron model
    perceptron_accuracy = digit_perceptron.predict(test_data_perceptron, testing_labels, perceptron_weights, perceptron_biases)
    print("PERCEPTRON: Testing Completed")
    print(f"PERCEPTRON: Testing Accuracy: {perceptron_accuracy:.2f}%\n")

if __name__ == "__main__":
    main()
