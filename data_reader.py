import numpy as np  



def load_images_from_file(filepath):
    with open(filepath, 'r') as file:
        content = file.readlines()

    images = []
    index = 0

    while index < len(content):
        if content[index].strip():
            limit = index + 20 if index + 20 < len(content) else len(content)
            block = content[index:limit]
            index = limit
            image_array = np.array([list(line.strip('\n')) for line in block])
            images.append(image_array)
        else:
            index += 1

    return images





def create_one_hot_labels_from_file(filepath):

    with open(filepath, 'r') as file:
        content = file.readlines()

    one_hot_labels = []

    for item in content:
        position = int(item.strip())
        vector = np.zeros(10, dtype=int)
        vector[position] = 1
        one_hot_labels.append(vector)

    return one_hot_labels



def load_integer_labels_from_file(filepath):
    with open(filepath, 'r') as file:
        content = file.readlines()

    integer_labels = []

    for item in content:
        label = int(item.strip())
        integer_labels.append(label)

    return integer_labels




def transform_to_binary_values(char_array):
    binary_representation = np.zeros(char_array.shape, dtype=int)
    for i in range(char_array.shape[0]):
        for j in range(char_array.shape[1]):
            for k in range(char_array.shape[2]):
                binary_representation[i, j, k] = 1 if char_array[i, j, k] != ' ' else 0

    return binary_representation


def flatten_images(images):
    flattened_images = []
    for image in images:
        flattened_image = [bit for row in image for bit in row]
        flattened_images.append(flattened_image)
    return flattened_images


def read_images(file_name, size):
    images = []
    try:
        with open(file_name) as file:
            count = 0
            image = []
            for line in file:
                row = ['0' if c == ' ' else '1' for c in line]
                image.append(row)
                count += 1
                if count == size:
                    images.append(image)
                    image = []
                    count = 0
    except FileNotFoundError:
        print(f"Error: File not found - {file_name}")
    return images


def read_labels(file_name):
    labels = []
    try:
        with open(file_name) as file:
            labels = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: File not found - {file_name}")
    return labels



def load_data_and_labels(images_path, labels_path, size):
    images = read_images(images_path, size)
    labels = read_labels(labels_path)
    return images, labels 