import numpy as np  # Importing the numpy library as np

def load_images_from_file(filepath):
    # Open a file to read image data blocks
    with open(filepath, 'r') as file:
        content = file.readlines()

    images = []
    index = 0
    # Process each line to form image data arrays
    while index < len(content):
        if content[index].strip():
            limit = index + 20 if index + 20 < len(content) else len(content)
            block = content[index:limit]
            index = limit
            # Form a numpy array for each block of image data
            image_array = np.array([list(line.strip('\n')) for line in block])
            images.append(image_array)
        else:
            index += 1

    return images

def create_one_hot_labels_from_file(filepath):
    # Open a file to read and convert labels to one-hot encoded vectors
    with open(filepath, 'r') as file:
        content = file.readlines()

    one_hot_labels = []

    for item in content:
        position = int(item.strip())
        # Generate a one-hot vector for each label
        vector = np.zeros(10, dtype=int)
        vector[position] = 1
        one_hot_labels.append(vector)

    return one_hot_labels

def load_integer_labels_from_file(filepath):
    # Read integer labels from a file
    with open(filepath, 'r') as file:
        content = file.readlines()

    integer_labels = []

    for item in content:
        label = int(item.strip())
        integer_labels.append(label)

    return integer_labels

def transform_to_binary_values(char_array):
    # Convert a character array into a binary array based on non-space values
    binary_representation = np.zeros(char_array.shape, dtype=int)
    for i in range(char_array.shape[0]):
        for j in range(char_array.shape[1]):
            for k in range(char_array.shape[2]):
                binary_representation[i, j, k] = 1 if char_array[i, j, k] != ' ' else 0

    return binary_representation
