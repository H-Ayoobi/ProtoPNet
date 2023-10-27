import os
import numpy as np
from PIL import Image
import PIL
import sys
import matplotlib.pyplot as plt


def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)
# Set the path to the dataset
# dataset_path = "/Users/hayoobi/PycharmProjects/ProtoArgNet/CUB_200_2011"
dataset_path = "/data2/CUB200/CUB_200_2011"

# Set the path for saving the processed data
cropped_path = os.path.join(dataset_path, "cub200")
trained_cropped = os.path.join(cropped_path, "train")
test_cropped = os.path.join(cropped_path, "test")
makedir(cropped_path)
makedir(trained_cropped)
makedir(test_cropped)


# Read the image_class_labels.txt file
labels_file = os.path.join(dataset_path, "image_class_labels.txt")
with open(labels_file, 'r') as file:
    lines = file.readlines()
labels = [int(line.split()[1]) for line in lines]

# Read the images.txt to find image file names
images_file = os.path.join(dataset_path, "images.txt")
with open(images_file, 'r') as file:
    lines = file.readlines()
image_file_names = [line.split()[1] for line in lines]

# Read the train_test_split.txt
split_train_test = os.path.join(dataset_path, "train_test_split.txt")
with open(split_train_test, 'r') as file:
    lines = file.readlines()
train_or_test = [int(line.split()[1]) for line in lines]

split_train_test = os.path.join(dataset_path, "bounding_boxes.txt")
with open(split_train_test, 'r') as file:
    lines = file.readlines()
bounding_boxes = [list(map(float, line.split()[1:])) for line in lines]

# Read and resize images
image_folder = os.path.join(dataset_path, "images") # using not cropped images


progress_interval = 100
for idx in range(0, len(image_file_names)): # using cropped images
    hmargin = int(0.1 * bounding_boxes[idx][3])
    wmargin = int(0.1 * bounding_boxes[idx][2])
    x, y, w, h = bounding_boxes[idx]
    image_path = os.path.join(image_folder, image_file_names[idx])  # using cropped images
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    width, height = image.size
    # image_path = os.path.join(image_folder, image_file_name) # using not cropped images
    # image_cropped = image.crop((x, y, x + w, y + h))
    image_cropped = image
    cropped_image_path = os.path.join(cropped_path, image_file_names[idx])
    makedir("/".join(cropped_image_path.split("/")[:-1]))
    image_cropped.save(cropped_image_path)
    if train_or_test[idx] == 1:
        trained_cropped_image_path = os.path.join(trained_cropped, image_file_names[idx])
        makedir("/".join(trained_cropped_image_path.split("/")[:-1]))
        image_cropped.save(trained_cropped_image_path)
    else:
        test_cropped_image_path = os.path.join(test_cropped, image_file_names[idx])
        makedir("/".join(test_cropped_image_path.split("/")[:-1]))
        image_cropped.save(test_cropped_image_path)

    if (idx + 1) % progress_interval == 0:
        progress = ((idx + 1) / len(image_file_names)) * 100
        sys.stdout.write(f"\rProgress: {int(progress)}%")
        sys.stdout.flush()


print("\nThe images are cropped and splitted to train and test!")