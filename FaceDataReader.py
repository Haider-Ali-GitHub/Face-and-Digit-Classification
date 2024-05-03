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

    with open(filename, 'r') as file:
        for line in file:
            stripped_line = line.strip()

            # Check if the current image is complete
            if found_first_line and len(current_image) == 67:
                images.append(current_image)
                current_image = []          # Reset for the next image
                found_first_line = False    # Reset the flag

            # Start a new image if not currently gathering one and find a non-empty line
            if not found_first_line and stripped_line:
                found_first_line = True
                current_image.append(stripped_line)
            elif found_first_line:
                current_image.append(stripped_line)

    # Append any final image that may not have been appended in the loop
    if current_image and len(current_image) == 67:
        images.append(current_image)

    return images

result = read_label('data/facedata/facedatatestlabels')
print()
print("These are all of the labels for the faces:")
print(result)
print()
print("These are how many labels we have:", len(result))

result = read_image('data/facedata/facedatatest')
print()
print("These are all of the images:")
#print(result)
print()
print("These are how many images we have:", len(result))

