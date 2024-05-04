import math

class NaiveBayesClassifier:
    """
    Naive Bayes classifier for face data.
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.k = 1  # Smoothing parameter
        self.featureCounts = {str(label): {} for label in legalLabels}
        self.labelCounts = {str(label): 0 for label in legalLabels}
        self.totalData = 0

    def train(self, trainingData, trainingLabels):
        total_items = len(trainingData)
        print(f"Starting training on {total_items} items...")
        for index, (datum, label) in enumerate(zip(trainingData, trainingLabels)):
            label = str(label)  # Convert label to string to prevent type issues
            self.labelCounts[label] += 1
            for feature, value in datum.items():
                if feature not in self.featureCounts[label]:
                    self.featureCounts[label][feature] = {'True': 0, 'False': 0}
                if value > 0:
                    self.featureCounts[label][feature]['True'] += 1
                else:
                    self.featureCounts[label][feature]['False'] += 1
            # Print progress every 100 items processed or last item
            if (index + 1) % 100 == 0 or (index + 1) == total_items:
                print(f"Processed {index + 1}/{total_items} training items...")
        self.totalData = sum(self.labelCounts.values())
        print("Training complete.")


    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        """
        guesses = []
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guess = max(posterior, key=posterior.get)
            guesses.append(guess)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        """
        logJoint = {}
        for label in self.legalLabels:
            logJoint[label] = math.log(self.labelCounts[label] / self.totalData)
            for feature, value in datum.items():
                featureCounts = self.featureCounts[label].get(feature, {'True': 0, 'False': 0})
                trueCount = featureCounts['True'] + self.k
                falseCount = featureCounts['False'] + self.k
                if value > 0:
                    logJoint[label] += math.log(trueCount / (trueCount + falseCount))
                else:
                    logJoint[label] += math.log(falseCount / (trueCount + falseCount))
        return logJoint

from data_reader import load_data_and_labels, flatten_images

def main():
    # File paths for digit data
    train_images_path = 'data/digitdata/trainingimages'
    train_labels_path = 'data/digitdata/traininglabels'
    validation_images_path = 'data/digitdata/validationimages'
    validation_labels_path = 'data/digitdata/validationlabels'
    test_images_path = 'data/digitdata/testimages'
    test_labels_path = 'data/digitdata/testlabels'

    # Load and prepare training data
    train_images, train_labels = load_data_and_labels(train_images_path, train_labels_path, 28)
    train_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(train_images)]

    # Load and prepare validation data
    validation_images, validation_labels = load_data_and_labels(validation_images_path, validation_labels_path, 28)
    validation_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(validation_images)]

    # Load and prepare test data
    test_images, test_labels = load_data_and_labels(test_images_path, test_labels_path, 28)
    test_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(test_images)]

    # Initialize the Naive Bayes classifier with possible digit labels
    nb_classifier = NaiveBayesClassifier(legalLabels=[str(i) for i in range(10)])  # Labels are strings from '0' to '9'

    # Training the classifier
    print("Training Naive Bayes Classifier...")
    nb_classifier.train(train_data, train_labels)

    # Validation
    print("Validating Naive Bayes Classifier...")
    validation_predictions = nb_classifier.classify(validation_data)
    validation_accuracy = sum(int(pred == true) for pred, true in zip(validation_predictions, validation_labels)) / len(validation_labels)
    print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

    # Testing
    print("Testing Naive Bayes Classifier...")
    test_predictions = nb_classifier.classify(test_data)
    test_accuracy = sum(int(pred == true) for pred, true in zip(test_predictions, test_labels)) / len(test_labels)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()

