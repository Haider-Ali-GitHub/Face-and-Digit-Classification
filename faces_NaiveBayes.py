import math

class NaiveBayesClassifier:

    # Naive Bayes classifier for FACES.
 
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
            label = str(label)
            if label not in self.labelCounts:
                print(f"Unexpected label {label} found! Skipping data point.")
                continue
            self.labelCounts[label] += 1
            for feature, value in datum.items():
                if feature not in self.featureCounts[label]:
                    self.featureCounts[label][feature] = {'True': 0, 'False': 0}
                self.featureCounts[label][feature]['True' if value > 0 else 'False'] += 1
            if (index + 1) % 100 == 0 or (index + 1) == total_items:
                print(f"Processed {index + 1}/{total_items} training images...")
        self.totalData = sum(self.labelCounts.values())


    # classifying data based on the posterior distribution over labels.
    def classify(self, testData):
        guesses = []
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guess = max(posterior, key=posterior.get)
            guesses.append(guess)
        return guesses

    def calculateLogJointProbabilities(self, datum):
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




# TO RUN; FOR TESTING PURPOSES!!!

from data_reader import load_data_and_labels, flatten_images

def main():
    # file paths
    train_images_path = 'data/facedata/facedatatrain'
    train_labels_path = 'data/facedata/facedatatrainlabels'
    validation_images_path = 'data/facedata/facedatavalidation'
    validation_labels_path = 'data/facedata/facedatavalidationlabels'
    test_images_path = 'data/facedata/facedatatest'
    test_labels_path = 'data/facedata/facedatatestlabels'

    # load/process training data
    train_images, train_labels = load_data_and_labels(train_images_path, train_labels_path, 70)
    train_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(train_images)]

    # load/process validation data
    validation_images, validation_labels = load_data_and_labels(validation_images_path, validation_labels_path, 70)
    validation_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(validation_images)]

    # load/process test data
    test_images, test_labels = load_data_and_labels(test_images_path, test_labels_path, 70)
    test_data = [{i: int(pixel) for i, pixel in enumerate(image)} for image in flatten_images(test_images)]

    # initialize the Naive Bayes classifier with possible face labels
    nb_classifier = NaiveBayesClassifier(legalLabels=[str(i) for i in range(2)])  # assuming labels '0' and '1'

    # training
    print("Training Naive Bayes Classifier for Face Data...")
    nb_classifier.train(train_data, train_labels)

    # validation
    print("Validating Naive Bayes Classifier for Face Data...")
    validation_predictions = nb_classifier.classify(validation_data)
    validation_accuracy = sum(int(pred == true) for pred, true in zip(validation_predictions, validation_labels)) / len(validation_labels)
    print(f"Validation Accuracy for Face Data: {validation_accuracy * 100:.2f}%")

    # testing
    print("Testing Naive Bayes Classifier for Face Data...")
    test_predictions = nb_classifier.classify(test_data)
    test_accuracy = sum(int(pred == true) for pred, true in zip(test_predictions, test_labels)) / len(test_labels)
    print(f"Test Accuracy for Face Data: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
