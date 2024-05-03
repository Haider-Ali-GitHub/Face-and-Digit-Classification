# perceptron.py
# -------------
# Licensing Information: You are free to use and extend these projects for educational purposes.
# The Pacman AI projects were developed at UC Berkeley, by John DeNero and Dan Klein.
# More info at http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util

class Perceptron:
    """
    Perceptron classifier.
    
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {label: util.Counter() for label in legalLabels}

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels), "Weights and legal labels must match in length"
        self.weights = weights
        
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        """
        self.features = list(trainingData[0].keys())  # could be useful later
        
        for iteration in range(self.max_iterations):
            print(f"Starting iteration {iteration}...")
            for i in range(len(trainingData)):
                actual, guess = trainingLabels[i], self.classify([trainingData[i]])[0]
                if actual != guess:
                    self.weights[actual] = self.weights[actual] + trainingData[i]
                    self.weights[guess] = self.weights[guess] - trainingData[i]
    
    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for label in self.legalLabels:
                vectors[label] = self.weights[label] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label.
        """
        featuresWeights = []
        # Example of defining high weight features -- needs real implementation
        if label in self.weights:
            featuresWeights = self.weights[label].most_common(100)  # Assuming Counter supports most_common

        return featuresWeights
