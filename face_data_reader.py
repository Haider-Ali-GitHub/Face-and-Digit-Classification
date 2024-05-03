def read_label(filename):
    """
    returns an array of all of the labels for the images
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def read_image(filename):
    """
    Reads and returns a list of images represented as lists of lines.
    Assumes each image starts with a non-empty line, followed by 66 more lines.

    Args:
        filename: The path to the image file.

    Returns:
        A list of images, where each image is a list of lines.
    """
    images = []
    current_image = []
    found_first_line = False
    size = 68  # Total lines per image including the starting line
    line_count = 0  # Count actual lines processed for each image

    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not found_first_line:
                if stripped_line:  # Start new image on first non-empty line
                    found_first_line = True
                    current_image.append(stripped_line)
                    line_count = 1
            else:
                current_image.append(stripped_line)
                line_count += 1
                if line_count == size:  # Check if current image has reached its expected size
                    images.append(current_image)
                    current_image = []
                    found_first_line = False
                    line_count = 0
                    print(f"Image added at line {line_number}, total images: {len(images)}")

    # Handle the last image in the file
    if current_image and line_count == size:
        images.append(current_image)
        print(f"Final image added at line {line_number}, total images: {len(images)}")

    return images
print()
print("TEST LABELS AND IMAGES")
result = read_label('data/facedata/facedatatestlabels')
print("These are how many labels we have:", len(result))
result = read_image('data/facedata/facedatatest')
print("These are how many images we have:", len(result))
print("-------------------------------------------------")

print()
print("TRAIN LABELS AND IMAGES")
result = read_label('data/facedata/facedatatrainlabels')
print("These are how many labels we have:", len(result))
result = read_image('data/facedata/facedatatrain')
print("These are how many images we have:", len(result))
print("-------------------------------------------------")

print()
print("VALIDATION LABELS AND IMAGES")
result = read_label('data/facedata/facedatavalidationlabels')
print("These are how many labels we have:", len(result))
result = read_image('data/facedata/facedatavalidation')
print("These are how many images we have:", len(result))

