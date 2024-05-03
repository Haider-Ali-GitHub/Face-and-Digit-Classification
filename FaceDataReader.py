def read_label(filename):
    """
    returns an array of all of the labels for the images
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def read_image(filename):
    """
    returns an array filled with arrays that have images inside of them 
    """
    images = []
    current_image = []
    blank_line_count = 0  # Counter for consecutive blank lines~!

    with open(filename, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                if blank_line_count >= 2 and current_image:  # Check if two blank lines were seen before this line
                    images.append(current_image)
                    current_image = []
                current_image.append(stripped_line)
                blank_line_count = 0  # Reset blank line counter
            else:
                blank_line_count += 1  # Increment blank line counter

        if current_image:  # Add the last image if file doesn't end with two blank lines
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

