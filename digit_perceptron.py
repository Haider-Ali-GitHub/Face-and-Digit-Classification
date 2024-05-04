import numpy as np
from digit_data_reader import convert_file_to_images, convert_labels_to_one_hot, convert_data_to_binary, convert_labels_to_data


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def train_perceptron(training_dataset, labels, epochs, lr):
    weights = np.random.rand(560, 10) * 0.01  # weights for 20x28 flattened array and 10 outputs
    bias = np.zeros((10,))  # biases for 10 classes

    for i in range(epochs):
        for training_image in range(len(training_dataset)):
            training_flat = training_dataset[training_image].reshape(-1, 560)
            output = softmax(np.dot(training_flat, weights) + bias)
            if labels[training_image].ndim == 1:
                error = labels[training_image] - output
            else:
                error = np.zeros((10,))
                error[labels[training_image]] = 1 - output 
            weights += lr * np.outer(training_flat, error)
            bias += lr * np.squeeze(error) 
    return weights, bias

def predict(validation_images, validation_labels, weights, bias):
    predictions = []
    for i in range(len(validation_images)):
        validation_flat = validation_images[i].reshape(-1, 560)
        probability = softmax(np.dot(validation_flat, weights) + bias)
        prediction = np.argmax(probability)
        predictions.append(prediction)
    correct_predictions = np.sum(predictions == validation_labels)
    total_predictions = len(validation_images)
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def main():
    # Paths for the training and validation data
    training_images_path = "data/digitdata/trainingimages"
    training_labels_path = "data/digitdata/traininglabels"
    validation_images_path = "data/digitdata/validationimages"
    validation_labels_path = "data/digitdata/validationlabels"
    test_data_path = "data/digitdata/testimages"
    test_labels_path = "data/digitdata/testlabels"

    # Loading the training data
    training_data = np.array(convert_file_to_images(training_images_path))
    training_labels = np.array(convert_labels_to_one_hot(training_labels_path))
    training_data = convert_data_to_binary(training_data)
    
    # Training the perceptron
    weights, bias = train_perceptron(training_data, training_labels, 5, .01)
    print("TRAINING COMPLETED")

    # Loading the validation data
    validation_data = np.array(convert_file_to_images(validation_images_path))
    validation_data = convert_data_to_binary(validation_data)
    validation_labels = np.array(convert_labels_to_data(validation_labels_path))
    
    # Predicting and printing validation accuracy
    print("VALIDATION COMPLETED")
    accuracy = predict(validation_data, validation_labels, weights, bias)
    print(f"Validation Accuracy: {accuracy}%")

if __name__ == "__main__":
    main()