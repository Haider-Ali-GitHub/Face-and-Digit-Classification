import numpy as np

def convert_file_to_images(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    arrays = [] 
    i = 0
    while i < len(lines):
        if lines[i].strip():  
            if i + 20 < len(lines):
                block = lines[i:i+20]
                i += 20  
            else:
                block = lines[i:]
                i = len(lines)
            array = np.array([list(line.rstrip('\n')) for line in block])
            arrays.append(array)
        else:
            i += 1

    return arrays

def convert_labels_to_one_hot(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    arrays = []

    for line in lines:
        index = int(line.strip())
        one_hot_vector = np.zeros(10, dtype=int)
        one_hot_vector[index] = 1
        arrays.append(one_hot_vector)

    return arrays

def convert_labels_to_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    arrays = []

    for line in lines:
        index = int(line.strip())
        arrays.append(index)

    return arrays

def convert_data_to_binary(array):
    binary_array = np.zeros(array.shape, dtype=int)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                if array[i, j, k] != ' ':
                    binary_array[i, j, k] = 1
    return binary_array