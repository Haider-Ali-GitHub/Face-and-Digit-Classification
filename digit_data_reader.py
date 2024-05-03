def load_images(file_path, lines_per_digit=20):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Normalize each line to the maximum length and segment into digit blocks
    digit_blocks = []
    current_block = []
    
    for line in lines:
        # Right pad lines with spaces to the maximum length
        normalized_line = line.rstrip().ljust(max_length)
        if line.strip():  # Non-empty line
            current_block.append(normalized_line)
            if len(current_block) == lines_per_digit:
                digit_blocks.append(current_block)
                current_block = []

    # Convert digit blocks to binary format
    return ascii_to_binary(digit_blocks)

def load_labels(file_path):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        labels = file.readlines()
        
    # Remove whitespace and return labels
    return [label.strip() for label in labels]

# Load the training images and labels
training_images_binary = load_images(training_images_path)
training_labels = load_labels(training_labels_path)

# Show some examples to verify
training_images_binary[0], training_labels[:10]
