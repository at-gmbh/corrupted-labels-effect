import math
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import preprocessing


def load_labels(npy_path):
    """
    Loads the npy-file dictionary serialization as produced by Generate_Data.py
    and reformats
      from dict[str => list[dict[str => Any]]]
      to dict[str => dict[str => Any]]

    Parameters
    ----------
    npy_path : str or path-like
        Path to the .npy file location to load from.

    Returns
    -------
    TYPE
        dict[dict[str => Any]].

    """
    labels_dict = np.load(npy_path, allow_pickle=True).item()

    return reformat_labels(labels_dict)



def select_label(labels_dict: dict, task: str):
    """
    Select labels for model task and flatten dict
        from from dict[str => dict[str => Any]]
        to dict[str => any]

    Parameters
    ----------
    labels_dict : dict[str => dict[str => Any]]
        Dictionary of dictionaries with multiple key-value pairs
    task : str
        Type of labels to be selected for model task
        'Class': labels for main class classification
        'Subclass': labels for sub class classification
        'Annotations' : labels for polygon vertices prediction

    Returns
    -------
    new_dict : dict[str => any]
        Ordinary dictionary with key-value pairs 
    """
    new_dict = dict()
    for image_id, list_of_dicts in labels_dict.items():
        if task in ['Class', 'Annotation']:
            new_dict[image_id] = list_of_dicts[task]
        elif task == 'Subclass':
            new_dict[image_id] = [list_of_dicts['Class'], list_of_dicts['Subclass']]
        else:
            print("Selected model task labels not existing.")
            sys.exit()
    
    return new_dict


def encode_labels(train_labels_dict: dict, test_labels_dict:dict):
    """
    Encode non-numerical labels to numerical labels

    Parameters
    ----------
    train_labels_dict / test_labels_dict : dict[str => str]
        Dictionary of observation ID and categorical label

    Returns
    -------
    encoded_dicts : list[dicts]
        list of new_dicts with numerical encoded labels 
        new_dict : dict[str => int]
            Dictionary of observation ID and numerical label
    """
    encoded_dicts = []
    le = preprocessing.LabelEncoder()
    
    # fit Encoder to labels from test and train to include edge cases
    y = list(train_labels_dict.values()) + list(test_labels_dict.values())
    le.fit(y)

    # transform labels
    for labels_dict in [train_labels_dict, test_labels_dict]:
        new_dict = {}
        for key, value in labels_dict.items():
            new_dict[key] = le.transform([value])[0]

        encoded_dicts.append(new_dict)

    return encoded_dicts


def reformat_labels(labels_dict):
    """
    Simplify a dictionary
      from dict[str => list[dict[str => Any]]]
      to dict[str => dict[str => Any]]

    Parameters
    ----------
    labels_dict : dict[str => list[dict[str => Any]]]
        Dictionary of lists of dictionaries (each with one key-value pair).

    Returns
    -------
    new_dict : dict[str => dict[str => Any]]
        Dictionary where each item is one dictionary containing all key-value
        pairs from the inner list of input dictionaries.

    """
    new_dict = dict()
    for image_id, list_of_dicts in labels_dict.items():
        new_dict[image_id] = dict()
        for property_dict in list_of_dicts:
            for key, value in property_dict.items():
                new_dict[image_id][key] = value
    return new_dict


def resize(image: Image, labels_dict: dict, new_size: tuple):
    """
    Resize a pillow image with its annotated polygon.

    "Annotations" must be a key in labels_dict.

    Parameters
    ----------
    image : PIL.Image
        The pillow image to be resized.
    labels_dict : dict
        The labels dictionary from the .npy labels file with type
        dict[str => dict[str => Any]].
    new_size : tuple
        2-tuple, (width, height) of the new image size.

    Returns
    -------
    img_res : resized pillow image
        Use np.array(img_res) to get the numpy array.
    labels_ret : dict
        labels dict with rescaled Annotations polygon and the tuple used for
        scaling stored in key "Scale". Use the latter to scale a prediction
        for this image back up into its original format.

    """
    img_res = image.resize(new_size)
    labels_ret = labels_dict.copy()

    orig_size = image.size
    scale = np.array(new_size) / np.array(orig_size)

    labels_ret["Annotations"] = labels_ret["Annotations"] * scale
    labels_ret["Scale"] = scale

    return img_res, labels_ret


def train_val_split(x_train, y_train, val_split):
    """
    Randomly split train img / labels into train img/ labels and
    validation img / labels

    Parameters
    ----------
    x_train : list
        list of img ids currently allocated for training and not testing
    y_train : dict[str => int]

    val_split : float
        split ration for validation
        0.2 => 20% of training data is removed from training used for validation
        Resulting number of val imgs / labels will be rounded down (math.floor)

    Returns
    -------
    x_train_splitted : list
        filtered list of img ids for training
    y_train_splitted : dict[str => int]
        filtered dict of img ids and corresponding label for training
    x_val : list
        list of img ids for validation
    y_val : dict[str => int]
        dict of img ids and corresponding label for validation
    """
    # from train img randomly select img for validation
    x_val = random.sample(list(x_train), math.floor(len(x_train) * val_split))
    # remove val img from train img
    x_train_splitted = [id for id in x_train if id not in x_val]

    y_val = {}
    y_train_splitted = y_train.copy()

    for id in x_val:
        y_val[id] = y_train[id]
        y_train_splitted.pop(id)
    
    return x_train_splitted, y_train_splitted, x_val, y_val

def make_false_labels(labels, false_labels_ratio, classes):
    """
    Randomly changes the label values in labels according to false_labels_ratio.

    Parameters
    ----------
    labels : dict[ str => str ]
        true labels
    false_labels_ratio : float
        Ratio of labels to make false between 0 and 1
        Resulting number of labels will be rounded down (math.floor)
    
    Returns
    -------
        numpy array with wrong labels
    Sample:
        print(y_train)
        >>> {'Train_00001': 0,
        >>>  'Train_00002': 0,
        >>>  'Train_00003': 2,
        >>>  'Train_00004': 0,
        >>>  'Train_00005': 2}
        print(make_false_labels(y_train, 0.5))
        >>> {'Train_00001': 0,
        >>>  'Train_00002': 0,
        >>>  'Train_00003': 2,
        >>>  'Train_00004': 3,
        >>>  'Train_00005': 4}
    """
    false_labels = labels
    n_labels = len(labels)

    # Generate the IDs that will have false label
    false_labels_count = math.floor(n_labels * false_labels_ratio)
    print("Number of false labels:", false_labels_count)
    wrong_label_keys = random.sample(list(labels.keys()), false_labels_count)

    # Manipulate label if index in wrong_label_ids
    for key in wrong_label_keys:
        # init wrong label
        wrong_label = None

        # get true value
        true_value = labels[key]

        # change label until label is wrong
        while wrong_label == None or wrong_label == labels[key]:
            wrong_label = random.randint(0, classes)

        # set wrong label
        false_labels[key] = wrong_label
    
    return false_labels


def plot_sample(image: np.ndarray, labels: dict):
    """
    Plot an image with its labels for debugging / inspecting.

    labels must contain keys "Img_id", "Class", "Subclass" and (optionally)
    "Annotations". The first three should be strings, the last a list of 4 2D
    points. The annotations are the polygon, this is only drawn when available.

    Parameters
    ----------
    image : numpy.ndarray
        An image as numpy array, plotted with plt.imshow.
    labels : dict
        The labels to be visualized in the plot.

    Returns
    -------
    None.

    """

    plt.imshow(image)

    plt.title(
        labels["Img_id"] + ": " + labels["Class"] + " - " + str(labels["Subclass"])
    )

    if "Annotations" in labels:
        line = plt.Polygon(labels["Annotations"], closed=True, fill=None, edgecolor="r")
        plt.gca().add_line(line)

    plt.show()
