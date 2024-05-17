# Face and Digit Classification Project

**Course:** CS 440: Introduction to Artificial Intelligence  
**Authors:** Diego Quito, Haider Ali, Steven Packard  
**Date:** May 1, 2024

## Introduction

In this project, we developed and analyzed three machine learning classifiers: a two-layer Neural Network, a Perceptron, and Naive Bayes, for the tasks of handwritten digit recognition and face detection using images pre-processed by our image reader. We systematically evaluated the performance of these classifiers across varying training sizes to determine the impact of training volume on classification accuracy and computational efficiency. This report details our methods, presents our experimental results, discusses the efficiency and accuracy of the implemented classifiers, and what we have learned along the way.

## Methodology

### Data Preparation

We initialized our project by defining file paths to our training, validation, and test datasets. For preprocessing, images are loaded using our `data reader` program, which scans through the datasets, stores them into memory, and transforms them into binary representations to reduce computational complexity. Digit images are standardized to 560 pixels (28x20) and face images to 1400 pixels (70x70), ensuring uniform input sizes for the models.

### Feature Design

Our system primarily uses raw pixels for its features. This allows our models to engage directly with the unmodified image data, facilitating direct pattern learning from the visual inputs. Given the low-resolution, dual-colored nature of the images, using raw pixels for features is efficient and effective.

### Testing

To test the entire system, we developed a file called `main.py`, which processes all datasets and executes all three classifiers on the processed image data. The terminal displays training time, running time, epoch number, loss (if available), validation accuracy, and testing accuracy for all classifiers. We also included `face_tester.py` and `digit_tester.py` executables for individual testing of the perceptron and neural network on the respective datasets.

## Classifier Descriptions

### Perceptron Classifier

The perceptron model consists of a single layer of weights plus a bias term. It is trained using an iterative update rule where weights are adjusted based on prediction errors on the training data. The learning rate parameter controls the magnitude of weight updates.

### Two-Layer Neural Network

#### Face Recognition Neural Network

Structured with an input layer for the 1400-pixel face images, a hidden layer using ReLU activation, and an output layer applying the softmax function for multi-class prediction. Training uses backpropagation to update weights and biases based on the gradient of the cross-entropy loss function.

#### Digit Recognition Neural Network

Similar to the face recognition network, but structured for 560-pixel digit images. It also includes a hidden layer with ReLU activation and an output layer utilizing softmax.

### Naive Bayes Classifier

This classifier operates using Bayes' Theorem, calculating the likelihood of a label based on the independent contributions of image features. Each pixel is treated as an independent feature. The classifier computes prior probabilities and pixel likelihoods during training and multiplies these during classification to produce posterior probabilities for each class.

## Experiment Setup

### Training

We designed a script that changes the training dataset size by 10% increments from 10% to 100%, allowing us to monitor scalability and learning efficiency. Each classifier was trained over a predefined number of epochs.

### Validation and Testing

Validation was conducted to tune hyperparameters and implement early stopping mechanisms. The final testing phase used the testing dataset to gauge the models' generalization ability.

## Results

### Performance Analysis

- **Perceptron:** Improved from 65.33% accuracy at 10% data to 89.33% at full data.
- **Neural Network:** Improved from 58.67% accuracy at 10% data to 90.00% at full data.
- **Naive Bayes:** Stable performance, ranging from 76.67% to 90.67% accuracy.

### Comparison Between Classifiers

| % Training Data | Perceptron | Neural Network | Naive Bayes |
|-----------------|------------|----------------|-------------|
| 10%             | 65.33%     | 58.67%         | 76.67%      |
| 20%             | 83.33%     | 77.33%         | 80.00%      |
| 30%             | 81.33%     | 80.67%         | 84.67%      |
| 40%             | 87.33%     | 83.33%         | 89.33%      |
| 50%             | 86.00%     | 87.33%         | 87.33%      |
| 60%             | 85.33%     | 89.33%         | 88.00%      |
| 70%             | 89.33%     | 88.00%         | 89.33%      |
| 80%             | 87.33%     | 89.33%         | 88.67%      |
| 90%             | 90.00%     | 90.00%         | 89.33%      |
| 100%            | 89.33%     | 90.00%         | 90.67%      |

### Statistical Analysis

- **Perceptron:** Average accuracy of 84.56%, standard deviation of 8.74%.
- **Neural Network:** Average accuracy of 84.33%, standard deviation of 10.45%.
- **Naive Bayes:** Average accuracy of 86.40%, standard deviation of 4.13%.

## Conclusion

This project provided hands-on experience with machine learning techniques for classifying handwritten digits and facial recognition. Our analysis highlighted the importance of larger training datasets in enhancing model performance. The neural network showed substantial improvements with more data, while the Naive Bayes classifier demonstrated consistent performance even with smaller datasets.

## References

- Python Software Foundation. (n.d.). Python language reference. Retrieved 2024, from https://www.python.org
- Cohen, P. R. (1995). Empirical methods for artificial intelligence. MIT Press.
- Klein, D., & DeNero, J. (2011). Project 5: Classification. CS188, Introduction to Artificial Intelligence, UC Berkeley. Retrieved 2024, from https://inst.eecs.berkeley.edu/~cs188/sp11/projects/classification/classification.html
