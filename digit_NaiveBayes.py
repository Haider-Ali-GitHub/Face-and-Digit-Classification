import math
import time

# NAIVE BAYES FOR FACE DATA
class NaiveBayesClassifier:
    
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.k = 1  # #smoothing
        self.featureCounts = {str(label): {} for label in legalLabels}
        self.labelCounts = {str(label): 0 for label in legalLabels}
        self.totalData = 0

    def train(self, trainingData, trainingLabels):
        total_items = len(trainingData)
        print(f"Starting training...")
        for index, (datum, label) in enumerate(zip(trainingData, trainingLabels)):
            label = str(label) 
            self.labelCounts[label] += 1
            for feature, value in datum.items():
                if feature not in self.featureCounts[label]:
                    self.featureCounts[label][feature] = {'True': 0, 'False': 0}
                if value > 0:
                    self.featureCounts[label][feature]['True'] += 1
                else:
                    self.featureCounts[label][feature]['False'] += 1
        self.totalData = sum(self.labelCounts.values())


    def classify(self, testData):
        # classify the data based on the posterior distribution over labels.
        guesses = []
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guess = max(posterior, key=posterior.get)
            guesses.append(guess)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        # returns the log-joint distribution over legal labels and the datum.
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




# FOR TESTING
def main():
    # file paths
    train_images_path = 'data/digitdata/trainingimages'
    train_labels_path = 'data/digitdata/traininglabels'
    validation_images_path = 'data/digitdata/validationimages'
    validation_labels_path = 'data/digitdata/validationlabels'
    test_images_path = 'data/digitdata/testimages'
    test_labels_path = 'data/digitdata/testlabels'

    # load/process training data
    
    train_images, train_labels = load_data_and_labels(train_images_path, train_labels_path, 28)
    train_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(train_images)]

    # load/process validation data
    validation_images, validation_labels = load_data_and_labels(validation_images_path, validation_labels_path, 28)
    validation_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(validation_images)]

    # load/process test data
    test_images, test_labels = load_data_and_labels(test_images_path, test_labels_path, 28)
    test_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(test_images)]

    # initialize the Naive Bayes classifier with possible digit labels
    nb_classifier = NaiveBayesClassifier(legalLabels=[str(i) for i in range(10)])  # labels are strings from '0' to '9'

    # training
    print("Training Naive Bayes Classifier...")
    first_time = time.time()
    nb_classifier.train(train_data, train_labels)
    training_time = time.time() - first_time
    print(f"Naive Bayes: Training Completed  (Training time: {training_time:.2f})")

    # validation
    print("Validating Naive Bayes Classifier...")
    validation_predictions = nb_classifier.classify(validation_data)
    validation_accuracy = sum(int(pred == true) for pred, true in zip(validation_predictions, validation_labels)) / len(validation_labels)
    elapsed_time = time.time() - first_time
    print(f"Validation Accuracy: {validation_accuracy * 100:.2f}% (Elapsed time: {elapsed_time:.2f}s)\n")

    # testing
    print("Testing Naive Bayes Classifier...")
    test_predictions = nb_classifier.classify(test_data)
    test_accuracy = sum(int(pred == true) for pred, true in zip(test_predictions, test_labels)) / len(test_labels)
    elapsed_time = time.time() - first_time
    print(f"Test Accuracy: {test_accuracy * 100:.2f}% (Elapsed time: {elapsed_time:.2f}s)\n")

if __name__ == "__main__":
    main()

