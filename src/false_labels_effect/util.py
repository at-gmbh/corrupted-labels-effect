import logging
import sys
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import yaml
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

# TODO: delete?
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
        dict[str => dict[str=> Any]].
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
