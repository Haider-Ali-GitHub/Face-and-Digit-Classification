import numpy as np
from data_reader import flatten_images, load_data_and_labels

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def cross_entropy_loss(x2, x1):
    shape = x2.shape[0]
    x1 = np.clip(x1, 1e-15, 1 - 1e-15)
    loss = -np.sum(x2 * np.log(x1)) / shape
    return loss


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        self.weights_hidden_process = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1. / self.hidden_size)

        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def forward(self, inputs):
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_process = relu(self.hidden_input)
        self.output_input = np.dot(self.hidden_process, self.weights_hidden_process) + self.bias_output
        self.output_process = softmax(self.output_input)  # chang sigmoid to softmax
        return self.output_process

    def backpropagation(self, inputs, y_true, lr):
        d_output = self.output_process - y_true
        error_hidden = d_output.dot(self.weights_hidden_process.T)
        d_hidden = error_hidden * relu_derivative(self.hidden_process)

        self.weights_hidden_process -= self.hidden_process.T.dot(d_output) * lr
        self.bias_output -= np.sum(d_output, axis=0, keepdims=True) * lr
        self.weights_input_hidden -= inputs.T.dot(d_hidden) * lr
        self.bias_hidden -= np.sum(d_hidden, axis=0, keepdims=True) * lr

    def train(self, training_data, training_labels, lr, epochs, batch_size):
        n_samples = training_data.shape[0]
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            training_data_shuffled = training_data[indices]
            training_labels_shuffled = training_labels[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_inputs = training_data_shuffled[start_idx:end_idx]
                batch_y_true = training_labels_shuffled[start_idx:end_idx]

                self.forward(batch_inputs)
                loss = cross_entropy_loss(batch_y_true, self.output_process)
                self.backpropagation(batch_inputs, batch_y_true, lr)

            if epoch % 5 == 0:
                print(f'Epoch: {epoch} |  Loss: {loss:.5f}')

    def predict(self, inputs):
        output_probabilities = self.forward(inputs)
        return np.argmax(output_probabilities, axis=1)

def calculate_accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

