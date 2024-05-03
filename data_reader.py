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
                row = ['0' if c == ' ' else '1' for c in line.strip()]
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
