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
def flatten_images(images):
    """
    Flattens a list of images from 2D lists of binary strings to 1D lists.

    Args:
    images (list): A list of images, where each image is a list of lists of binary strings.

    Returns:
    list: A list of flattened images, where each image is represented as a single list of binary strings.
    """
    flattened_images = []
    for image in images:
        flattened_image = [bit for row in image for bit in row]
        flattened_images.append(flattened_image)
    return flattened_images

def read_images(file_name, size):
    """ 
    Reads image data from a file and converts each line of the file into a binary format based on spaces or marks.
    Each line is read into an array where spaces are converted to '0' and any non-space character is converted to '1'.

    Args:
    file_name (str): Path to the file containing image data.
    size (int): The height of the image in lines.

    Returns:
    list: A list of images, with each image represented as a list of binary strings.
    """
    images = []
    try:
        with open(file_name) as file:
            count = 0
            image = []
            for line in file:
                # Convert line to binary format while removing any excess spaces before and after non space characters 
                row = ['0' if c == ' ' else '1' for c in line]
                image.append(row)
                count += 1
                # Check if current image is complete
                if count == size:
                    images.append(image)
                    image = []
                    count = 0
    except FileNotFoundError:
        print(f"Error: File not found - {file_name}")
    return images

def read_labels(file_name):
    """
    Reads labels from a file. Each label is assumed to be on a new line.
    
    Args:
    file_name (str): Path to the file containing labels.
    
    Returns:
    list: A list of labels.
    """
    labels = []
    try:
        with open(file_name) as file:
            # Read each non-empty line as a label
            labels = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: File not found - {file_name}")
    return labels

def load_data_and_labels(images_path, labels_path, size):
    """
    Loads both images and labels using the specified paths and logs the process.
    
    Args:
    images_path (str): Path to the image data file.
    labels_path (str): Path to the label data file.
    size (int): The height of each image in lines.
    
    Returns:
    tuple: A tuple containing two lists - one for images and one for labels.
    """
    images = read_images(images_path, size)
    labels = read_labels(labels_path)
    return images, labels

