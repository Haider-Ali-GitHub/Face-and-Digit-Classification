class Perceptron:
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {label: {} for label in legalLabels}

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        for iteration in range(self.max_iterations):
            print(f"Starting iteration {iteration}...")
            for features, true_label in zip(trainingData, trainingLabels):
                # Compute the dot product of weights and features for each label
                scores = {label: sum(features[f] * self.weights[label].get(f, 0) for f in features) for label in self.legalLabels}
                # Determine the best guess label
                best_guess_label = max(scores, key=scores.get)
                # Update weights if the prediction is wrong
                if true_label != best_guess_label:
                    for f in features:
                        if f in self.weights[true_label]:
                            self.weights[true_label][f] += features[f]
                        else:
                            self.weights[true_label][f] = features[f]
                        
                        if f in self.weights[best_guess_label]:
                            self.weights[best_guess_label][f] -= features[f]
                        else:
                            self.weights[best_guess_label][f] = -features[f]

    def test(self, data):
        guesses = []
        for datum in data:
            scores = {label: sum(datum[f] * self.weights[label].get(f, 0) for f in datum) for label in self.legalLabels}
            best_label = max(scores, key=scores.get)
            guesses.append(best_label)
        return guesses

    # def findHighWeightFeatures(self, label, top_k=100):
    #     if label in self.weights:
    #         # Get features with the highest weights for the given label
    #         features_weights = list(self.weights[label].items())
    #         # Sort by weight in descending order
    #         features_weights.sort(key=lambda x: x[1], reverse=True)
    #         # Return the top k features
    #         return features_weights[:top_k]
    #     else:
    #         return []

