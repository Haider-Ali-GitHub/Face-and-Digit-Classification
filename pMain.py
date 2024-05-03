from data_reader import read_images, read_labels
from perceptron import Perceptron


def main():
    # Define the path to the data files
    train_images_file = 'data/digitdata/trainingimages'
    train_labels_file = 'data/digitdata/traininglabels'
    test_images_file = 'data/digitdata/testimages'
    test_labels_file = 'data/digitdata/testlabels'
    validation_images_file = 'data/digitdata/validationimages'
    validation_labels_file = 'data/digitdata/validationlabels'
    
    # Assuming each image size is 28x28 (this needs to be adjusted according to actual data format)
    image_size = 28
    
    # Read training and testing data
    training_images = read_images(train_images_file, image_size)
    training_labels = read_labels(train_labels_file)
    testing_images = read_images(test_images_file, image_size)
    testing_labels = read_labels(test_labels_file)
    validation_images = read_images(validation_images_file, image_size)
    validation_labels = read_labels(validation_labels_file)

    # Create a Perceptron classifier
    # For digit recognition, labels might be 0-9, adjust as necessary for your specific task
    legal_labels = range(10)  # Change if doing face/not face to something like [0, 1]
    perceptron = Perceptron(legalLabels=legal_labels, max_iterations=3)  # You can adjust the number of iterations

    # Train the Perceptron
    perceptron.train(training_images, training_labels, validation_images, validation_labels)

    # Test the Perceptron
    predictions = perceptron.test(testing_images)

    # Print out the accuracy or any other performance metric
    correct_count = sum(1 for pred, label in zip(predictions, testing_labels) if pred == label)
    accuracy = correct_count / len(testing_labels)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
