import numpy as np
from data_reader import load_images_from_file, create_one_hot_labels_from_file, transform_to_binary_values, load_integer_labels_from_file

def softmax(x):
    # Apply softmax function to the input array.
    exponentiated = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)

def train_perceptron(training_data, target_labels, num_epochs, learning_rate):
    # Train a perceptron model with the specified parameters. 
    weights = np.random.rand(560, 10) * 0.01  # Initial weights for a 20x28 flattened input to 10 output classes
    biases = np.zeros((1, 10))  # Initialize biases for 10 classes

    for epoch in range(num_epochs):
        cumulative_loss = 0
        for index in range(len(training_data)):
            flattened_input = training_data[index].reshape(-1, 560)
            prediction = softmax(np.dot(flattened_input, weights) + biases)
            if target_labels[index].ndim == 1:
                error = target_labels[index] - prediction
            else:
                error = np.zeros((1, 10))
                error[0, target_labels[index]] = 1 - prediction
            weights += learning_rate * np.dot(flattened_input.T, error)
            biases += learning_rate * error
            cumulative_loss += np.sum(-target_labels[index] * np.log(prediction + 1e-15)) / target_labels.shape[0]

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {cumulative_loss:.4f}')

    return weights, biases

def predict(validation_set, validation_labels, weights, biases):
    # Make predictions using the trained perceptron model. 
    predicted_labels = []
    for image in range(len(validation_set)):
        flattened_image = validation_set[image].reshape(-1, 560)
        probabilities = softmax(np.dot(flattened_image, weights) + biases)
        predicted_label = np.argmax(probabilities)
        predicted_labels.append(predicted_label)
    accuracy = np.mean(predicted_labels == validation_labels) * 100
    return accuracy

def main():
    # Define file paths for data
    training_images_path = "data/digitdata/trainingimages"
    training_labels_path = "data/digitdata/traininglabels"
    validation_images_path = "data/digitdata/validationimages"
    validation_labels_path = "data/digitdata/validationlabels"

    # Load data and preprocess
    training_data = np.array(load_images_from_file(training_images_path))
    training_labels = np.array(create_one_hot_labels_from_file(training_labels_path))
    training_data = transform_to_binary_values(training_data)

    # Execute training process
    weights, biases = train_perceptron(training_data, training_labels, 100, 0.01)
    print("TRAINING COMPLETED")

    # Load validation data and transform
    validation_data = np.array(load_images_from_file(validation_images_path))
    validation_data = transform_to_binary_values(validation_data)
    validation_labels = np.array(load_integer_labels_from_file(validation_labels_path))

    # Validate the model and print results
    accuracy = predict(validation_data, validation_labels, weights, biases)
    print("VALIDATION COMPLETED")
    print(f"Validation Accuracy: {accuracy}%")

if __name__ == "__main__":
    main()

