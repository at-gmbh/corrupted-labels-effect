# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:00:31 2020

@author: Wolfgang Reuter

This script sets up the altered images and respective labels for the project.

There are four main alteration classes:
    - Blob:             Adds blobs to the image within a randomly generated
                        polygonal or rectangular shape
    - Blur:             Blurs the image within a randomly generated polygonal
                        or rectangular shape
    - Distortion:       Randomly adds or subtracts a value in a specified
                        range, i.e. [50 -> 100] to each pixel in a randomly
                        generated polygonal or rectangular shape. If the
                        value of 255 is exceeded, the new value gets
                        replaced by 255 and if the resulting value is
                        negative, it gets replaced by 0.
    - Channel_change:   Changes the order of the color channel

There are also subclasses for the main classes Blob, Distortion and
Channel_Change, representing the color of the blobs (Red, Green, Blue or
All), the Channel of the distortion (Red, Green, Blue or All) or the new
order of the channels ([Red, Blue, Green], [Blue, Red, Green],
[Blue, Green, Red], [Green, Red, Blue] or [Green, Blue, Red]).

The labels are saved as a dictionary with the image filename as key,

i.e. Train_00011,

and a list of dictionaries as value, i.e.

[{'Img_id': 'Train_00011'},
 {'Class': 'Blob'},
 {'Subclass': 'Green'},
 {'Annotations': [[38, 64], [328, 117], [242, 273], [119, 238], [38, 64]]}]

The labels dictionaries are stored in .npy format in the specified folders.

IMPORTANT NOTE:
    The target directories for images and labels and the subdirectories for
    the images, i.e. Train, Val and Test, have to be set up before this
    script is run

TODO: Alter so that directories are created if they are not already there

IMPORTANT NOTE:
    The altered images can either be stored in RGB or BGR channel order
    in the specified folders - according to the preference of the user.
    They are saved in .png (lossless) format.

"""

# =============================================================================
# Imports
# =============================================================================

import os
import sys

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import util
from tqdm import tqdm, trange

# =============================================================================
# Seeds
# =============================================================================

# Make sure to generate images in same order
np.random.seed(1007)

# =============================================================================
# Paths and Variables
# =============================================================================

# Save data, set to False for testing
ready_to_save = True

# When h5-files are available, define whether training or test dataset shall be loaded,
# use "Train" or "Test"
set_name = 'Test'

# When h5-files are not available, use the raw_dir setting to supply a location
# containing raw jpg files, this location will be treated as Training data
raw_dir = os.path.join("data", "Raw")

# Choose whether to generate polygons or, if set to False, rectangles
generate_polys = True

# If you want to save your images in rgb format, set this to True
store_images_as_rgb = True

# For testing
num_to_display = 0

# Load the images into RAM
# NOTE: Currently NOT loading images into memory is not supported
load_images_into_memory = True

# Show loading progress
verbose = True

# Parameters to restrict generated ramdom polygons (i.e. no very near points,
# and no near straight lines)
vicinity_th = 1 / 10
angle_range = [140, 220]

# Define a list of image alteration classes
alterations_4_classes_list = ["Blob", "Blur", "Distortion", "Channel_Change"]

# Parameters for alterations

# BLOB
# Minimum and maximum number of blobs generated (scattered over the
# whole image, not only the polygon area)
n_blobs_min = 300
n_blobs_max = 500

# Minimum and maximum radius of the blobs generated
min_radius = 1
max_radius = 7

# Minimum value of randomly chosen pixel value
min_pixel_val = 100

# BLUR
# Minimum and maximum value either taken away or added to pixels in
# a color channel
ksize_h = 35
ksize_w = 35

# DISTORTION
# Minimum and maximum value either taken away or added to pixels in
# a color channel
min_distortion = 50
max_distortion = 100

# List of valid channel names
channel_display_list = ["Red", "Green", "Blue", "All"]
color_display_list = channel_display_list
channel_name_arr = np.array(channel_display_list[:3])

# Path to the PacalVoc Datasets
# TODO: Replace with the directory where the images you want to distort are stored
hdf5_train_path = ".\\data\\dataset_pascal_voc_07+12_trainval.h5"
hdf5_test_path = ".\\data\\dataset_pascal_voc_07_test.h5"

# Target paths
if generate_polys == True:
    poly_substring = "_Poly"
else:
    poly_substring = "_Rect"

# TODO: Replace with the directory where the distorted images are to be
#       stored
if "set_name" in locals():
    images_target_path = os.path.join("data", "Images_4c" + poly_substring, set_name)
    labels_target_path = os.path.join("data", "Images_4c" + poly_substring, set_name)
else:
    images_target_path = os.path.join("data", "Images_4c" + poly_substring, "Train")
    labels_target_path = os.path.join("data", "Images_4c" + poly_substring, "Train")

if not os.path.exists(images_target_path):
    os.makedirs(images_target_path)

# =============================================================================
# Load image data
# =============================================================================

if "set_name" in locals():  # keep old functionality with h5 input
    if set_name == "Train":
        hdf5_dataset_path = hdf5_train_path
    elif set_name == "Test":
        hdf5_dataset_path = hdf5_test_path
    else:
        print("You need to specify training or test set!")

    hdf5_dataset = h5py.File(hdf5_dataset_path)
    dataset_size = len(hdf5_dataset["images"])
    dataset_indices = np.arange(dataset_size, dtype=np.int32)

    if load_images_into_memory:
        images = []
        if verbose:
            tr = trange(
                dataset_size, desc="Loading images into memory", file=sys.stdout
            )
        else:
            tr = range(dataset_size)
        for i in tr:
            images.append(
                hdf5_dataset["images"][i].reshape(hdf5_dataset["image_shapes"][i])
            )
else:  # raw_dir must be set
    images, sizes = list(), list()
    min_width, min_height = np.inf, np.inf

    for image_filename in tqdm(os.listdir(raw_dir)):
        image_path = os.path.join(raw_dir, image_filename)
        image = util.load_image(image_path)
        images.append(image)

        size = image.shape[:2]
        if size not in sizes:
            sizes.append(size)
        min_width, min_height = min(min_width, size[1]), min(min_height, size[0])

    print(f"{len(images)} images loaded")
    print(f"Distinct image sizes: {sizes}")
    print(f"Images have at least width = {min_width} and height = {min_height}")

# =============================================================================
# Alter images and Set up labels
# =============================================================================

# Labels dictionary
labels_dict = {}

# Set up a counter to generate img_ids in the form set_name_00001,
# i.e. Train_00001
string_index = 1

# For testing
display_inds = set(np.random.choice(len(images), num_to_display))

if "set_name" in locals():
    prefix = set_name
else:
    prefix = "Train"

for i in tqdm(range(len(images))):
    # Get temporary image with width and height
    img = images[i]

    w = img.shape[1]
    h = img.shape[0]

    # Set up image id (to be used as filename and key in labels dictionary)
    img_id = prefix + "_" + str(string_index).zfill(5)

    # Generate list of lists of polygonal (or rectangular) coordinates
    if generate_polys:
        poly_coord = util.generate_random_poly(
            w,
            h,
            check_vicinity=True,
            check_near_straight_line=True,
            vicinity_th=vicinity_th,
            angle_range=angle_range,
        )

    else:
        poly_coord = util.generate_random_rect(
            w, h, check_vicinity=True, vicinity_th=vicinity_th
        )

    # Randomly choose a distortion and its predefined name
    class_index = np.random.randint(len(alterations_4_classes_list))
    class_name = alterations_4_classes_list[class_index]

    # Get mask (black image)
    mask = np.zeros_like(img)

    # Set additional class information to None
    # Only used to print out correct additional labels
    color_channel_index = None
    channel_index = None
    index_arr = None

    # Alter image within polygon according to chosen class
    if class_name == "Blob":
        new_img, color_channel_index = util.blob(img, poly_coord, mask)
        subclass = color_display_list[color_channel_index]
    elif class_name == "Blur":
        new_img = util.blur(img, poly_coord, mask, ksize_h, ksize_w)
        subclass = None
    elif class_name == "Distortion":
        new_img, channel_index = util.distort(
            img, poly_coord, mask, min_distortion, max_distortion
        )
        subclass = channel_display_list[channel_index]
    elif class_name == "Channel_Change":
        new_img, index_arr = util.channel_change(img, poly_coord, mask)
        subclass = str(list(channel_name_arr[index_arr]))
    else:
        print("\nYou need to stick to the predefined class names!")
        assert 1 == 0

    # Set up labels
    labels_dict[img_id] = [
        {"Img_id": img_id},
        {"Class": class_name},
        {"Subclass": subclass},
        {"Annotations": poly_coord},
    ]

    # Display randomly chosen images
    if i in display_inds:
        xs, ys = zip(*poly_coord)  # create lists of x and y values

        print("\nAlteration class: ", alterations_4_classes_list[class_index])
        if color_channel_index != None:
            print("Color: ", subclass)
        if channel_index != None:
            print("Channel: ", subclass)
        if np.all(index_arr) != None:
            print("Order: ", subclass)

        fig = plt.figure(figsize=(18, 7))

        fig.add_subplot(1, 4, 1)
        plt.imshow(img)

        fig.add_subplot(1, 4, 2)
        plt.imshow(img)
        plt.plot(xs, ys, c="r", linewidth=3)

        fig.add_subplot(1, 4, 3)
        plt.imshow(new_img)
        plt.plot(xs, ys, c="r", linewidth=3)

        fig.add_subplot(1, 4, 4)
        plt.imshow(new_img)

        plt.show()

    # Store images either to RGB or BGR color channel order
    if store_images_as_rgb:
        new_img = new_img.astype("uint8")
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    else:
        new_img = new_img.astype("uint8")

    if ready_to_save:
        if not cv2.imwrite(os.path.join(images_target_path, img_id + ".png"), new_img):
            raise Exception("Could not write image")

    string_index += 1

if ready_to_save:
    np.save(labels_target_path, labels_dict)
