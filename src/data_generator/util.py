import logging
import math
from pathlib import Path
from typing import Any, Dict, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import yaml
from matplotlib.patches import Circle
from PIL import Image

"""
Created on Sat Aug 29 11:16:03 2020

@author: Wolfgang Reuter
"""
def load_image(path, resize=None):
    """
    Load an image from a file location directly into numpy.

    Resizing is optional but possible.
    Note, that if you need to resize the polygon label in case you are
    predicting bounding boxes, you should load the images with pillow
    directly and then use the false-labels-dlb.util.resize function.

    Parameters
    ----------
    path : str or path-like
        Path to the image file location.
    resize : tuple, optional
        If not None, then it resizes the image into (width, height).
        The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    try:
        image = Image.open(path)

        if resize is not None:
            image = image.resize(resize)

        return np.array(image)
    except TypeError as te:
        print(te)


def generate_random_poly(
    img_width,
    img_height,
    check_vicinity=False,
    check_near_straight_line=False,
    vicinity_th=1.0 / 10,
    angle_range=[175, 185],
):
    """
    Generates a random, convex polygon within the dimensions of an image,
    so that the polygon points are ordered clockwise and the leftmost
    point is the first point.

    If check_vicinity is set to True:
        Only polygons with a specified minimum distance between any two
        points are returned.

    if check_near_straight_line is set to True:
        Only polygons with inner angles between any two lines outside a
        specified angle_range are returned.
    """

    # Helper boolean to avoid concave polygons
    correct_true_label = False

    # Generate polygon points randomly until they are convex
    while not correct_true_label:

        # Empty list of polygon points
        tl_coord = []
        for i in range(4):
            px = np.random.randint(img_width)
            py = np.random.randint(img_height)

            tl_coord.append([px, py])

        # close polygon
        tl_coord.append(tl_coord[0])

        # Order points
        tl_coord = order(tl_coord)

        # Check if points are convex
        if concave(tl_coord) == False:
            correct_true_label = True

        #        print(vicinity(img_width, img_height, tl_coord, 1./10))

        # Check for nearby points
        if check_vicinity and point_vicinity(
            img_width, img_height, tl_coord, vicinity_th
        ):
            correct_true_label = False

        if check_near_straight_line and near_straight_line(
            tl_coord, angle_range=angle_range
        ):

            correct_true_label = False

    return tl_coord


def generate_random_rect(
    img_width, img_height, check_vicinity=False, vicinity_th=1.0 / 10
):
    """
    Generates a rectangle within the dimensions of an image,
    so that the points are ordered clockwise and the leftmost
    point is the first point.

    NOTE: Each rectancle point is represented as a list of two entries,
          i.e. the x- and y-coordinate of the point. This is not an ideal
          representation of a rectangle - but currently used for consistency,
          as the rest of the script(s) in which it is used is set up so that
          convex polygons can be visualized.

    If check_vicinity is set to True:
        Only polygons with a specified minimum distance between any two
        points are returned.
    """

    # Helper boolean to avoid concave polygons
    correct_true_label = False

    # Generate polygon points randomly until they are convex
    while not correct_true_label:

        # Empty list of polygon points
        tl_coord = []
        for i in range(2):
            px = np.random.randint(img_width)
            py = np.random.randint(img_height)

            tl_coord.append([px, py])
        tl_coord_arr = np.array(tl_coord)

        x1 = np.min(tl_coord_arr[:, 0])
        x2 = np.max(tl_coord_arr[:, 0])
        y1 = np.min(tl_coord_arr[:, 1])
        y2 = np.max(tl_coord_arr[:, 1])

        tl_coord = []
        tl_coord.append([x1, y1])
        tl_coord.append([x2, y1])
        tl_coord.append([x2, y2])
        tl_coord.append([x1, y2])

        # close polygon
        tl_coord.append(tl_coord[0])

        # Order points
        tl_coord = order(tl_coord)

        if len(tl_coord) != 5:
            continue

        # Check if points are convex
        if concave(tl_coord) == False:
            correct_true_label = True

        #        print(vicinity(img_width, img_height, tl_coord, 1./10))

        # Check for nearby points
        if check_vicinity and point_vicinity(
            img_width, img_height, tl_coord, vicinity_th
        ):
            correct_true_label = False

    return tl_coord


def order(point_list):
    """
    Orders the points of a polygon according to the following restrictions:
        - Leftmost point is first
        - If there are two leftmost points (with same x-coordinate),
          the upper one is first
        - All other points follow ordered in clockwise direction

    Mathematical logic:

        This function takes in a list of polygon points and sorts them
        with respect to x-values in ascending order and also with respect
        to y-values in ascending order.

        The first point is extracted - which is the leftmost point
        (and the upper one in case the leftmost point is not unique).

        From the first point, it calclates the angles
        to all other points and puts the points in such order that
        the angles are sorted.

    NOTE: Throws an error if enpty list is passed in. For empty
          intersections, i.e. polygons that do NOT intersect, this
          is handled in intersection_polygon, which then also returns
          an empty list. In addition, in poly_area a try-except
          clause has been added, which will return 0 if area can not
          be calculated.

    """

    # Convert to numpy array
    point_arr = np.array(point_list)

    # Get unique values of point arr
    point_arr = np.unique(point_arr, axis=0)

    # Get indices with respect to ascending x-values and, in
    # case of equality, with respect to ascending y-values
    arr_inds = np.lexsort((point_arr[:, 1], point_arr[:, 0]))

    # Sort according to ascending x-values and, in case of equality,
    # with respect to ascending y-values
    point_arr = point_arr[arr_inds]

    # Get first point
    first_point = [point_arr[0][0], point_arr[0][1]]

    # Calculate tangens to all other points
    tan_arr = np.arctan2(
        point_arr[1:, 1] - first_point[1], point_arr[1:, 0] - first_point[0]
    )

    # Get indices of sorted point arr without first point
    # with respect to tan_arr
    inds = np.argsort(tan_arr, axis=0)

    # Sort poins 2 to last with respect to tan_arr and convert to list
    second_to_last_point_list = list(point_arr[inds + 1].tolist())

    # Put first point in a list of lists and add other points as list
    return [first_point] + second_to_last_point_list + [first_point]


def concave(polygon):
    """
    Returns true if a 4-edged polygon is convex (in computer image
    representation, i.e. upper left point is 0,0, NOT lower left point).
    Polygon is a list of lists, whereby the first inner list entry contain
    the x and y values of the first point, the second the ones of the
    second point, etc.

    Mathematical logic:

        A polygon in computer vision representation is convex, if ALL
        possible traversals, i.e. ABC, BCD, CDA, DAB - are either clockwise
        or counter clockwise - but not mixed. Due to our SSD representation,
        where all polygons are orientated clockwise, only clockwise
        orientation is checked.
    """
    concave = False
    for a in range(4):
        clockwise = ccw(polygon[a % 4], polygon[(a + 1) % 4], polygon[(a + 2) % 4])
        if clockwise == False:
            concave = True
    return concave


def ccw(A, B, C):
    """
    Returns True if three points A, B, C are aligned anticlockwise,
    False otherwise.

    Mathematical logic:

        Imagine three points A, B, C. The points, if traversed from A via B
        to C, are aligned clockwise if and only if the slope of the line
        AB is bigger then the slope of the line AC. Otherwise they are
        aligned anticlockwise (or on a straight line, a case that is neither
        clockwise nor anticlockwise).

        In an x-y-coordinate system the slope of AB can be expressed as
        the difference of the y-values of B and A divided by the difference
        of the x-values of B and A, i.e. (By-Ay) / (Bx-Ax)

        Similarly the slope of AC is calculated as (Cy-Ay) / (Cx-Ax)

        Clockwise orientation hence requires:

            (By-Ay) / (Bx-Ax) > (Cy-Ay) / (Cx-Ax)

        to be true. Multiplied out you get the equation below.

    NOTE: In our case the function returns the opposite value, as height is
    measured from 0 to height of image DOWNWARDS, i.e. in the opposite
    direction as in an ordinary x-y-coordinate system
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def point_vicinity(img_width, img_height, poly, vicinity_th):
    """
    Takes the minimum dimension (min_dim) of an image and calculates a
    theshold (th) given a specified fraction (vicinity_th) of the
    minimum dimension. If any two points of a polygon are closer to each
    other than the threshold (th), the function returns True,
    False otherwise.

    Used to omitt generated polygons with any two points beeing very
    close together (as tey are similar to triangles).
    """
    min_dim = np.minimum(img_width, img_height)
    th = min_dim * vicinity_th

    if poly[0] == poly[-1]:
        poly = poly[:-1]

    distances = []

    for i in range(len(poly)):
        for j in range(i + 1, len(poly)):
            d = np.sqrt(
                np.square(poly[i][0] - poly[j][0]) + np.square(poly[i][1] - poly[j][1])
            )

            distances.append(d)

    if np.min(np.array(distances)) < th:
        return True
    else:
        return False


def near_straight_line(poly, angle_range=[178, 182]):
    """
    Checks whether any two adjacent lines in a polygon are within a
    specified angle_range.

    Used to omitt generated polygons with "nearly" straight lines
    between any two points (as they are similar to triangles)
    """

    l = len(poly)

    straight_line = False

    for i in range(l):
        p1 = poly[i % l]
        p2 = poly[(i + 1) % l]
        p3 = poly[(i + 2) % l]

        if (
            get_angle(p1, p2, p3) > angle_range[0]
            and get_angle(p1, p2, p3) < angle_range[1]
        ):
            straight_line = True
    return straight_line


def get_angle(a, b, c):
    """
    Returns the inner angle of points a, b and c within a polygon. The angle
    is measured at point b.

    Mathematical logic:

        https://en.wikipedia.org/wiki/Atan2

        atan2 measures the angle between the positive x-axis and a point
        (x,y).

        get_angle therefore measures the angle between the positive x-axis
        and c by mapping b on the origin of the positive x-axis (i.e. taking
        the difference in y-values of c and b and also the difference in
        x-values of c and b).

        The angle between the positive x-axis and a is measured accordingly.

        These two angles are then subtracted and transformed to degrees.
        If the resulting angle is negative, 360 degrees are added.

        This gibes the outer (positive) angle between a and c at point b.
        To get the inner angle, the outer angle is subtracted from
        360 degrees.


    """
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    ang = ang + 360 if ang < 0 else ang
    return 360 - ang


# =============================================================================
# Functions to set up the alterations of the images and associated labels
# =============================================================================


def distort(img, poly_coord, mask, min_distortion, max_distortion):
    """
    Distorts an image within the bounds of a convex polygon (or rectangle)
    in the following way:
        1) A color chanel, within which the pixels are distorted,
           is randomly chosen - or all color channels are distorted.
        2) A random integer is chosen between min_distorion and
           max_distortion and added or subtracted to each pixel within the
           bounds of the polygon
        3) Resulting pixels with values bigger than 255 or smaller than zero
           are set to 255 or zero respectivley

    Parameters
    ----------
    img : NUMPY ARRAY
        An image with three color channels
    poly_coord : LIST OF LISTS
        A list with the lists of x and y coordinates of the corners of a
        convex, closed polygon (or rectangle), with the upper leftmost
        point being the first in the list.
    mask : NUMPY ARRAY
         A numpy array of zeros in the shape of the image
    min_distortion : INTEGER
         The minimum value added to or subtracted from a pixel
    max_distortion : INTEGER
         The maximum value added to or subtracted from a pixel

    Returns
    -------
    The distorted image and the color channel(s) which were distorted

    """
    a = cv2.fillConvexPoly(mask, np.array(poly_coord), (255, 255, 255))

    inds = np.where(a == 255)

    new_img = img.copy()
    new_img = new_img.astype(np.int32)

    distortion_channel = np.random.randint(4)

    if distortion_channel < 3:
        new_img[inds[0], inds[1], distortion_channel] += np.random.choice(
            [-1, 1]
        ) * np.random.randint(min_distortion, max_distortion)

    else:
        new_img[inds] += np.random.choice([-1, 1]) * np.random.randint(
            min_distortion, max_distortion
        )

    max_inds = np.where(new_img > 255)
    new_img[max_inds] = 255
    min_inds = np.where(new_img < 0)
    new_img[min_inds] = 0

    assert np.sum(np.where(new_img > 255)) == 0
    assert np.sum(np.where(new_img < 0)) == 0

    return new_img, distortion_channel


def channel_change(img, poly_coord, mask):
    """
    Changes the order of the color channels of an image
    within the bounds of a convex polygon (or rectangle)

    Parameters
    ----------
    img : NUMPY ARRAY
        An image with three color channels
    poly_coord : LIST OF LISTS
        A list with the lists of x and y coordinates of the corners of a
        convex, closed polygon (or rectangle), with the upper leftmost
        point being the first in the list.
    mask : NUMPY ARRAY
         A numpy array of zeros in the shape of the image
    Returns
    -------
    The distorted image

    """
    a = cv2.fillConvexPoly(mask, np.array(poly_coord), (255, 255, 255))

    inds = np.where(a == 255)

    new_img = img.copy()

    r_channel = new_img[:, :, 0]
    g_channel = new_img[:, :, 1]
    b_channel = new_img[:, :, 2]

    channel_list = [r_channel, g_channel, b_channel]

    index_arr = np.array([0, 1, 2])
    old_index_arr = index_arr.copy()

    same_arr = True

    while same_arr:
        np.random.shuffle(index_arr)
        if (
            index_arr[0] != old_index_arr[0]
            or index_arr[1] != old_index_arr[1]
            or index_arr[2] != old_index_arr[2]
        ):
            same_arr = False

    color_x_img = np.zeros(img.shape)
    color_x_img[:, :, 0] = channel_list[index_arr[0]]
    color_x_img[:, :, 1] = channel_list[index_arr[1]]
    color_x_img[:, :, 2] = channel_list[index_arr[2]]

    new_img[inds] = color_x_img[inds]

    return new_img, index_arr


def blur(img, poly_coord, mask, ksize_h, ksize_w):
    """
    Blurs an image within the bounds of a convex polygon (or rectangle)

    Parameters
    ----------
    img : NUMPY ARRAY
        An image with three color channels
    poly_coord : LIST OF LISTS
        A list with the lists of x and y coordinates of the corners of a
        convex, closed polygon (or rectangle), with the upper leftmost
        point being the first in the list.
    mask : NUMPY ARRAY
         A numpy array of zeros in the shape of the image
    ksize_h : INTEGER
         The height of the kernel used in the cv2 bluring function
    ksize_w : INTEGER
         The width of the kernel used in the cv2 bluring function

    Returns
    -------
    The distorted image

    """
    a = cv2.fillConvexPoly(mask, np.array(poly_coord), (255, 255, 255))

    inds = np.where(a == 255)

    new_img = img.copy()

    blurred_img = cv2.GaussianBlur(new_img, (ksize_h, ksize_w), 0)

    new_img[inds] = blurred_img[inds]

    assert np.sum(np.where(new_img > 255)) == 0
    assert np.sum(np.where(new_img < 0)) == 0

    return new_img


def blob(
    img,
    poly_coord,
    mask,
    n_blobs_min=300,
    n_blobs_max=500,
    min_radius=1,
    max_radius=7,
    min_pixel_val=100,
):
    """
    Adds blobs on an image within the bounds of a convex polygon
    (or rectangle)

    Parameters
    ----------
    img : NUMPY ARRAY
        An image with three color channels
    poly_coord : LIST OF LISTS
        A list with the lists of x and y coordinates of the corners of a
        convex, closed polygon (or rectangle), with the upper leftmost
        point being the first in the list.
    mask : NUMPY ARRAY
         A numpy array of zeros in the shape of the image
    n_blobs_min : INTEGER
         Minimum number of blobs to be generated (and scattered over the
         whole image, NOT only with in the polygonal area)
    n_blobs_max : INTEGER
         Maximum number of blobs to be generated (and scattered over the
         whole image, NOT only with in the polygonal area)
    min_radius : INTEGER
         Minimum size of the radius of a blob, in pixels
    max_radius : INTEGER
         Maximum size of the radius of a blob, in pixels
    min_pixel_val : INTEGER
         Minimum value of a randomly generated blob color

    Returns
    -------
    The distorted image

    """

    w = img.shape[1]
    h = img.shape[0]

    a = cv2.fillConvexPoly(mask, np.array(poly_coord), (255, 255, 255))

    inds = np.where(a == 255)

    n_blobs = np.random.randint(n_blobs_min, n_blobs_max)

    # Make some example data
    x = np.random.randint(max_radius, w - max_radius, n_blobs)
    y = np.random.randint(max_radius, h - max_radius, n_blobs)

    # Create a figure. Equal aspect so circles look circular
    fig, ax = plt.subplots(1)
    # ax.set_aspect('equal')

    # Show the image
    ax.imshow(img)
    ax.axis("off")

    fig = plt.gcf()
    fig.add_axes(ax)
    DPI = fig.get_dpi()
    fig.set_size_inches(w / float(DPI), h / float(DPI))

    color_list = [0, 0, 0]

    color_channel_index = np.random.randint(4)

    # Now, loop through coord arrays, and create a circle at each x,y pair
    for xx, yy in zip(x, y):

        if color_channel_index < 3:
            color_list[color_channel_index] = (
                np.random.randint(min_pixel_val, 255) / 255.0
            )

        elif color_channel_index == 3:
            color_list[0] = np.random.randint(min_pixel_val, 255) / 255.0
            color_list[1] = np.random.randint(min_pixel_val, 255) / 255.0
            color_list[2] = np.random.randint(min_pixel_val, 255) / 255.0

        circ = Circle(
            (xx, yy), np.random.randint(min_radius, max_radius), color=tuple(color_list)
        )
        ax.add_patch(circ)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("temp_image.png", bbox_inches="tight", pad_inches=0)

    plt.close()

    blobed_img = cv2.imread("temp_image.png")

    blobed_img = cv2.resize(blobed_img, (w, h))
    blobed_img = cv2.cvtColor(blobed_img, cv2.COLOR_BGR2RGB)

    new_img = img.copy()
    new_img[inds] = blobed_img[inds]

    assert np.sum(np.where(new_img > 255)) == 0
    assert np.sum(np.where(new_img < 0)) == 0

    return new_img, color_channel_index
