from data_reader import read_data, read_labels
from perceptron import Perceptron

def main():
    # Initialize the perceptron
    legalLabels = list(range(10))  # Adjust according to your labels, e.g., range(10) for digits, range(2) for face/no-face
    max_iterations = 100
    classifier = Perceptron(legalLabels, max_iterations)

    # Load data
    train_images = read_data('data/digitdata/trainingimages', 28)
    train_labels = read_labels('data/digitdata/traininglabels')
    test_images = read_data('data/digitdata/testimages', 28)
    test_labels = read_labels('data/digitdata/testlabels')

    # Train the classifier
    classifier.train(train_images, train_labels, [], [])

    # Test the classifier
    accuracy = classifier.classify(test_images, test_labels)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()