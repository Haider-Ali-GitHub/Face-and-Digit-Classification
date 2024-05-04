import numpy as np
from digit_data_reader import load_images_from_file, create_one_hot_labels_from_file, transform_to_binary_values, load_integer_labels_from_file


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
        self.output_process = sigmoid(self.output_input)
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
            traing_data_shuffled = training_data[indices]
            training_labels_shuffled = training_labels[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_inputs = traing_data_shuffled[start_idx:end_idx]
                batch_y_true = training_labels_shuffled[start_idx:end_idx]

                self.forward(batch_inputs)
                loss = cross_entropy_loss(batch_y_true, self.output_process)
                self.backpropagation(batch_inputs, batch_y_true, lr)

            if epoch % 10 == 0: 
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

def predict(validation_images, weights_input_hidden, weights_hidden_process, bias_hidden, bias_output):
    hidden_input = np.dot(validation_images, weights_input_hidden) + bias_hidden
    hidden_process = relu(hidden_input)

    final_input = np.dot(hidden_process, weights_hidden_process) + bias_output
    return softmax(final_input)

def main():
    input_size = 560 
    hidden_size = 300  
    output_size = 10  

    nn = NeuralNetwork(input_size, hidden_size, output_size)

    training_data_path = "data/digitdata/trainingimages"
    training_labels_path = "data/digitdata/traininglabels"
    validation_data_path = "data/digitdata/validationimages"
    validation_labels_path = "data/digitdata/validationlabels"
    test_data_path = "data/digitdata/testimages"
    test_labels_path = "data/digitdata/testlabels"

    training_data = np.array(load_images_from_file(training_data_path))
    training_labels = np.array(create_one_hot_labels_from_file(training_labels_path))
    training_data = transform_to_binary_values(training_data)
    training_data = np.reshape(training_data, (5000, 560))

    nn.train(training_data, training_labels, lr=0.001, epochs=100, batch_size=32)
    print("TRAINING COMPLETED")

    validation_data = np.array(load_images_from_file(validation_data_path))
    validation_data = transform_to_binary_values(validation_data)
    validation_labels = np.array(load_integer_labels_from_file(validation_labels_path))
    validation_data = np.reshape(validation_data, (1000, 560))
    print("VALIDATION COMPLETED")

    predicted_labels = predict(validation_data, nn.weights_input_hidden, nn.weights_hidden_process, nn.bias_hidden, nn.bias_output)
    accuracy = np.mean(np.argmax(predicted_labels, axis=1) == validation_labels)
    print(f"Validation Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
