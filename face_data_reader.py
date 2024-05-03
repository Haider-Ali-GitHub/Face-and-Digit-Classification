import numpy as np
from random import sample
import math
import time
import timeit
import matplotlib.pyplot as plt

def read_data(file_name, size):
    image_size = size
    num_images = 0
    numbers = []
    with open(file_name) as fp:
        line = fp.readline()
        count_rows = 0
        image = []
        while line:
            row = []
            for c in line:
                if c == ' ':
                    row.append('0')
                else:
                    row.append('1')
            row.pop(len(row) - 1)
            image.append(row)

            count_rows += 1
            if count_rows == image_size:
                count_rows = 0
                numbers.append(image)
                num_images += 1
                image = []
            line = fp.readline()
    return numbers

def read_labels(file_name):
    labels = []
    num_labels = 0
    with open(file_name) as fp:
        line = fp.readline()
        while line:
            line = line[:-1]
            labels.append(line)
            line = fp.readline()
            num_labels += 1
    return labels


# Example function usage
print("TEST LABELS AND IMAGES")
test_labels = read_labels('data/facedata/facedatatestlabels')
print("These are how many labels we have:", len(test_labels))
test_images = read_data('data/facedata/facedatatest', 70) 
print("These are how many images we have:", len(test_images))
print("-------------------------------------------------")

print("TRAIN LABELS AND IMAGES")
train_labels = read_labels('data/facedata/facedatatrainlabels')
print("These are how many labels we have:", len(train_labels))
train_images = read_data('data/facedata/facedatatrain', 70)  
print("These are how many images we have:", len(train_images))
print("-------------------------------------------------")

print("VALIDATION LABELS AND IMAGES")
validation_labels = read_labels('data/facedata/facedatavalidationlabels')
print("These are how many labels we have:", len(validation_labels))
validation_images = read_data('data/facedata/facedatavalidation', 70)  
print("These are how many images we have:", len(validation_images))
