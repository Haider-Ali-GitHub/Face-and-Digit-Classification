def read_label(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


result = read_label('data/facedata/facedatatestlabels')
print()
print("These are all of the labels for the faces:")
print(result)
print()
print("These are how many labels we have:", len(result))

