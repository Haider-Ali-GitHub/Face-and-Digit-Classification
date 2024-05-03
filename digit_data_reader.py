def load_images(file_path):
    """
    Loads images from a file and converts them to a list of 2D pixel representations.

    Args:
        file_path: Path to the image file.

    Returns:
        A list of images, where each image is a list of rows, and each row is a list of pixel values.
    """

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Calculate the maximum length of any line
    max_length = max(len(line.rstrip()) for line in lines if line.strip())

    # Normalize, segment into blocks, and handle potential errors
    digit_blocks = []
    current_block = []
    for line in lines:
        normalized_line = line.rstrip().ljust(max_length)

        # Error handling: Check for inconsistent lengths within a digit block
        if len(normalized_line) != max_length:
            raise ValueError(f"Inconsistent line length. Expected {max_length}, got {len(normalized_line)}")

        # Error handling: Ensure only valid characters (+, #, or space) are present
        if not all(char in ['+', '#', ' '] for char in normalized_line):
            raise ValueError(f"Unexpected characters in image data: {normalized_line}")

        if line.strip():  # Non-empty line
            current_block.append(normalized_line)
        else:  # Empty line signals the end of a digit block
            if current_block:  # Only append if the block has content
                digit_blocks.append(current_block)
                current_block = []


    return digit_blocks



def load_labels(file_path):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        labels = file.readlines()
        
    # Remove whitespace and return labels
    return [label.strip() for label in labels]

def ascii_to_binary(digit_blocks):
    binary_images = []
    for block in digit_blocks:
        binary_image = []
        for line in block:
            binary_line = [1 if char in ['+', '#'] else 0 for char in line]
            binary_image.append(binary_line)
        binary_images.append(binary_image)
    return binary_images


# Assume this is how you might call the functions after adjusting the paths to match your file system
result = load_images('data/digitdata/testimages')
print("Number of digit TEST images loaded:", len(result))

labels_result = load_labels('data/digitdata/testlabels')
print("Number of TEST labels loaded:", len(labels_result))

TrainImage = load_images('data/digitdata/trainingimages')
print("Number of digit TRAIN images loaded:", len(TrainImage))

TrainLabel = load_labels('data/digitdata/traininglabels')
print("Number of TRAIN labels loaded:", len(TrainLabel))

ValidationImage = load_images('data/digitdata/validationimages')
print("Number of digit VALIDATION images loaded:", len(ValidationImage))

ValidationLabel = load_labels('data/digitdata/validationlabels')
print("Number of VALIDATION labels loaded:", len(ValidationLabel))
