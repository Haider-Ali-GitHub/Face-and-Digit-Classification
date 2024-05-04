import numpy as np
from data_reader import load_images_from_file, create_one_hot_labels_from_file, transform_to_binary_values, load_integer_labels_from_file
import digit_neural
import digit_perceptron
import time

def main():

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
    start_time = time.time()
    print("Training Neural Network...")
    nn = digit_neural.NeuralNetwork(input_size=560, hidden_size=300, output_size=10)
    nn.train(training_data, training_labels, lr=0.001, epochs=100, batch_size=32)
    training_time = time.time() - start_time
    print(f"\nNEURAL NETWORK: Training Completed (Elapsed training time: {training_time:.2f}s)")

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
    elapsed_time = time.time() - start_time
    nn_accuracy = np.mean(np.argmax(nn_predicted_labels, axis=1) == validation_labels)
    print("NEURAL NETWORK Validation Completed")
    print(f"NEURAL NETWORK Validation Accuracy: {100*nn_accuracy:.2f}% (Elapsed time: {elapsed_time:.2f}s)\n")

    # testing the neural Network
    nn_predicted_labels = digit_neural.predict(testing_data, nn.weights_input_hidden, nn.weights_hidden_process, nn.bias_hidden, nn.bias_output)
    nn_accuracy = np.mean(np.argmax(nn_predicted_labels, axis=1) == testing_labels)
    elapsed_time = time.time() - start_time
    print("NEURAL NETWORK Testing Completed")
    print(f"NEURAL NETWORK Testing Accuracy: {100*nn_accuracy:.2f}% (Elapsed time: {elapsed_time:.2f}s)\n")

    


    print("\nPERCEPTRON- Digit")
    print("--------------------")

    # reuse training data for perceptron
    training_data_perceptron = transform_to_binary_values(np.array(load_images_from_file(training_images_path)))

    # perceptron training 
    start_time = time.time()
    print("Training Perceptron...")
    perceptron_weights, perceptron_biases = digit_perceptron.train_perceptron(training_data_perceptron, training_labels, 100, 0.01)
    training_time = time.time() - start_time
    print(f"\nPERCEPTRON: Training Completed (Elapsed training time: {training_time:.2f}s)")

    # load validation data for perceptron
    validation_data_perceptron = transform_to_binary_values(np.array(load_images_from_file(validation_images_path)))
    validation_data_perceptron = np.reshape(validation_data_perceptron, (len(validation_data_perceptron), 560))

    # validate perceptron
    
    perceptron_accuracy = digit_perceptron.predict(validation_data_perceptron, validation_labels, perceptron_weights, perceptron_biases)
    elapsed_time = time.time() - start_time
    print("PERCEPTRON: Validation Completed")
    print(f"PERCEPTRON: Validation Accuracy: {perceptron_accuracy:.2f}% (Elapsed time: {elapsed_time:.2f}s)\n")

    # load testing data for perceptron
    test_data_perceptron = transform_to_binary_values(np.array(load_images_from_file(test_images_path)))
    test_data_perceptron = np.reshape(test_data_perceptron, (len(test_data_perceptron), 560))

    # test the perceptron
    perceptron_accuracy = digit_perceptron.predict(test_data_perceptron, testing_labels, perceptron_weights, perceptron_biases)
    elapsed_time = time.time() - start_time
    print("PERCEPTRON: Testing Completed")
    print(f"PERCEPTRON: Testing Accuracy: {perceptron_accuracy:.2f}% (Elapsed time: {elapsed_time:.2f}s)\n")

if __name__ == "__main__":
    main()
