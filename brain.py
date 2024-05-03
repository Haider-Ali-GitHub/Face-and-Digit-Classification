inport
from perceptron import Perceptron

def flatten_image(image):
    """Flatten a 2D image into a 1D list of pixels."""
    return [pixel for row in image for pixel in row]

def setup_data_paths(classification_type):
    """Set up paths based on the type of classification chosen."""
    if classification_type == '1':
        data_type = 'facedata'
        labels = ('a face', 'not a face')
    elif classification_type == '2':
        data_type = 'digitdata'
        labels = ('the digit 1', 'not the digit 1')
    else:
        return None, None

    paths = {
        "train_images": f'{data_type}/{"facedatatrain" if classification_type == "1" else "trainingimages"}',
        "train_labels": f'{data_type}/{"facedatatrainlabels" if classification_type == "1" else "traininglabels"}',
        "test_images": f'{data_type}/{"facedatatest" if classification_type == "1" else "testimages"}',
        "test_labels": f'{data_type}/{"facedatatestlabels" if classification_type == "1" else "testlabels"}'
    }
    return paths, labels

def load_and_prepare_data(paths):
    """Load and prepare image data by flattening it."""
    train_images = load_images(paths['train_images'])
    train_labels = load_labels(paths['train_labels'])
    train_flattened_images = [flatten_image(image) for image in train_images]
    test_images = load_images(paths['test_images'])
    test_labels = load_labels(paths['test_labels'])
    test_flattened_images = [flatten_image(image) for image in test_images]
    return train_flattened_images, train_labels, test_flattened_images, test_labels

def train_and_test_perceptron(train_data, train_labels, test_data, test_labels, labels):
    """Train the perceptron and test it, printing results for each image."""
    num_features = len(train_data[0]) if train_data else 0
    perceptron = Perceptron(num_features)
    perceptron.train(train_data, train_labels, epochs=10)
    correct_predictions = 0
    for i, (image, label) in enumerate(zip(test_data, test_labels)):
        result = perceptron.predict(image)
        if result == label:
            correct_predictions += 1
        result_label = labels[0] if result == 1 else labels[1]
        print(f"Image {i+1} is {result_label}.")
    accuracy = correct_predictions / len(test_labels) * 100
    return accuracy

def main():
    """Main function to handle user input and initiate data handling and processing."""
    while True:
        print("Select the type of classification:")
        print("1: Faces")
        print("2: Digits")
        classification_type = input("Enter choice (1-2): ")
        if classification_type in ['1', '2']:
            break
        print("Invalid choice. Please select 1 or 2.")

    paths, labels = setup_data_paths(classification_type)
    if not paths:
        print("Failed to setup data paths. Exiting...")
        return

    train_data, train_labels, test_data, test_labels = load_and_prepare_data(paths)
    test_accuracy = train_and_test_perceptron(train_data, train_labels, test_data, test_labels, labels)
    print(f"Overall test accuracy: {test_accuracy:.2f}%")

if __name__ == '__main__':
    main()