import data_reader
import os
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
    print(f"Loading data from {images_path} and labels from {labels_path}")
    images = data_reader.read_images(images_path, size)
    labels = data_reader.read_labels(labels_path)
    print(f"Loaded {len(labels)} labels and {len(images)} images.")
    print("-------------------------------------------------")
    return images, labels

FACE_DIR = 'data/facedata'
DIGIT_DIR = 'data/digitdata'

FACE_IMAGE_SIZE = 70
DIGIT_IMAGE_SIZE = 28

face_datasets = {
    'test': ('facedatatest', 'facedatatestlabels'),
    'train': ('facedatatrain', 'facedatatrainlabels'),
    'validation': ('facedatavalidation', 'facedatavalidationlabels')
}

digit_datasets = {
    'test': ('testimages', 'testlabels'),
    'train': ('trainingimages', 'traininglabels'),
    'validation': ('validationimages', 'validationlabels')
}

for name, (img_file, label_file) in digit_datasets.items():
    images_path = os.path.join(DIGIT_DIR, img_file)
    labels_path = os.path.join(DIGIT_DIR, label_file)
    digit_images, digit_labels = load_data_and_labels(images_path, labels_path, DIGIT_IMAGE_SIZE)
    flat_digit_images = data_reader.flatten_images(digit_images)

for name, (img_file, label_file) in face_datasets.items():
    images_path = os.path.join(FACE_DIR, img_file)
    labels_path = os.path.join(FACE_DIR, label_file)
    face_images, face_labels = load_data_and_labels(images_path, labels_path, FACE_IMAGE_SIZE)
    flat_face_images = data_reader.flatten_images(face_images)
    print(flat_face_images)

