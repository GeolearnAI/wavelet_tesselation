# -*- coding: utf-8 -*-

"""Multiscale hierarchical domaining using continuous wavelets.

Copyright © Geolearn, Jerome Simon
https://github.com/GeolearnAI/wavelet_tesselation
"""

import time
from os import listdir

import pywt
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.color import rgb2grey, rgb2hsv


# ## Entire LEM-18 drillhole

# Concatenate image while converting and applying mean

onlyfiles = listdir('LEM-37\Export_10cm_slices')

im = np.empty([1000 * len(onlyfiles)], dtype=np.float32)

for i, image in enumerate(onlyfiles):
    temp = np.array(Image.open('LEM-37/Export_10cm_slices/' + image))
    temp = temp[:, 150:500, :]
    temp = rgb2grey(temp)
    temp = np.mean(temp, axis=1).reshape([-1, 1])
    im[i * 1000:(i + 1) * 1000] = temp

fig, axes = plt.subplots(1, 1, figsize=(1, 20))
axes.imshow(im, interpolation='none', aspect='auto', cmap=mpl.cm.gray)
plt.show()


def pad(image, pad_fraction=.2):
    """Pad an image to avoid border effects."""
    pad = int(pad_fraction * image.shape[0])
    image = np.hstack([image[pad:0:-1], im, image[-2:-pad - 2:-1]])
    return image


def compute_wavelets(image, scales):
    scales = np.logspace(2., 3., 5)

    coefs = pywt.cwt(data=im, scales=scales, wavelet='mexh')
    coefs = np.swapaxes(coefs[0], 0, 1)

    return coefs


def unpad(image, coefs, pad_fraction=.2):
    """Unpad a padded image."""
    pad = int(pad_fraction * image.shape[0])
    image = image[pad:-pad]
    pad = int(pad_fraction * coefs.shape[0] / (1+2*pad_fraction))
    coefs = coefs[pad:-pad]


def normalize_coefficients(coefs):
    """Normalize coefficients to maximum amplitude of 1."""
    coef_max = np.amax(coefs)
    coefs /= coef_max
    return coefs


def tesselate(coefs, filter_threshold=.15, min_width_filter=0):
    contours = get_contours(coefs)
    rectangles = get_rectangles_from_contours(coefs, min_width_filter)
    rectangles = delete_overlapping_rectangles(
        coefs,
        rectangles,
        filter_threshold,
    )
    rectangles = generate_children(rectangles)


def get_contours(coefs, plot=False):
    plt.figure(figsize=(3, 100))

    coef_max = np.amax(coefs)
    plt.imshow(
        coefs,
        interpolation='nearest',
        aspect='auto',
        cmap=mpl.cm.bwr,
        vmin=-coef_max,
        vmax=coef_max,
    )
    contours = plt.contour(coefs, levels=[0])

    if plot:
        plt.show()
    else:
        plt.clf()

    return contours


def get_rectangles_from_contours(coefs, contours, min_width_filter=0):
    paths = contours.collections[0].get_paths()
    rectangles = np.empty([len(paths), 4], dtype=np.int32)

    for i, path in enumerate(paths):
        v = path.vertices
        x = v[:, 0]
        y = v[:, 1]
        if y[0] > y[-1]:
            y[0], y[-1] = y[-1], y[0]
        # Rectangle format: [x, y, width, height].
        rectangles[i] = [np.amin(x), y[0], np.amax(x), y[-1] - y[0]]

    touches_left = (rectangles[:, 0] == 0)
    has_non_null_height = (rectangles[:, 3] != 0)
    has_minimum_width = (rectangles[:, 2] > min_width_filter)
    is_valid = touches_left & has_non_null_height & has_minimum_width
    rectangles = rectangles[is_valid]
    rectangles = np.vstack(
        [[0, 0, coefs.shape[1] - 1, coefs.shape[0] - 1], rectangles]
    )
    return rectangles


def delete_overlapping_rectangles(
            coefs,
            rectangles,
            filter_threshold=.15,
            verbose=True,
        ):
    mask = np.ones(rectangles.shape[0], dtype=bool)

    for i1, r1 in enumerate(rectangles):
        for i2, r2 in enumerate(rectangles):
            if r2[2] > r1[2] or i1 == i2 or not mask[i2]:
                continue
            upper_in = r2[1] >= r1[1] and r2[1] < r1[1] + r1[3]
            bottom_in = r2[1]+r2[3] > r1[1] and r2[1]+r2[3] <= r1[1]+r1[3]
            if upper_in != bottom_in:
                mask[i2] = False
    rectangles = rectangles[mask]

    # If a rectangle does not contain an abs(value) greater than
    # `filter_threshold`, it is deleted.
    mask = np.ones(rectangles.shape[0], dtype=bool)
    coef_max = np.amax(coefs)
    coef_max *= filter_threshold
    for i, r in enumerate(rectangles):
        rectangle_values = np.abs(coefs[r[1]:r[1]+r[3]+1, r[0]:r[0]+r[2]+1])
        if not (rectangle_values > coef_max).any():
            mask[i] = False
    initial_qty = rectangles.shape[0]
    rectangles = rectangles[mask]
    if verbose:
        print("Rectangles left:", rectangles.shape[0], '/', initial_qty)

    return rectangles


def generate_children(rectangles, verbose=True):
    if verbose:
        now = time.time()

    flag = True
    # Added vertical rectangles, the parents.
    ver_rect = []
    # Array which increase computation effectiveness.
    non_parents = np.zeros(rectangles.shape[0], dtype=bool)
    # The set of children doesn't need to be modified, as new children won't
    # be wider than the original children and as parents' `max_x` remain at
    # their original width. Also note that
    # `temp[n, 0] + temp [n, 2] == temp[n, 2]` for every `n`, as
    # `temp[n, 0] == 0`.
    temp = np.copy(rectangles)

    # While there are still horizontal rectangles that are added.
    while flag:
        flag = False
        # Keeps track of elements to delete after the loops.
        bool_ = np.ones(rectangles.shape[0], dtype=bool)
        # Added horizontal rectangles, the children.
        hor_rect = []

        for i1, r1 in enumerate(rectangles):
            if non_parents[i1]:  # If the rectangle does not have any child
                continue

            children = []

            for i2, r2 in enumerate(temp):

                if r2[1] >= r1[1] and r2[1] < r1[1] + r1[3] and r2[3] < r1[3] and r2[2] <= r1[2]:
                    # TODO Time consuming, add unique_id?
                    if not (r1 == r2).all():
                        children.append(i2)  # r2 is a children.

            if len(children) == 0:  # If there are no children.
                non_parents[i1] = True
                continue

            upper_x = [temp[i][2] for i in children]
            max_x = max(upper_x)

            # Get the eldest children.
            children = [temp[i] for i, j in zip(children, upper_x) if j == max_x]
            children.sort(key=lambda x: x[1])

            if r1[2] - max_x != 0:
                ver_rect += [(max_x, r1[1], r1[2] - max_x, r1[3])]
            bool_[i1] = False

            # Add borders.
            children = np.vstack(
                [
                    np.array([[0, r1[1], max_x, 0]]),
                    *children,
                    np.array([[0, r1[1] + r1[3], max_x, 0]])
                ]
            )

            # Add a rectangle between each pair of children.
            for top, bottom in zip(children[:-1], children[1:]):
                if bottom[1] - top[1] - top[3] != 0:
                    hor_rect += [(0, top[1]+top[3], max_x, bottom[1]-top[1]-top[3])]
                    flag = True

        rectangles = rectangles[bool_]
        # The new children could be parents; those must be iterated over.
        rectangles = np.vstack([rectangles, *hor_rect])

        non_parents = non_parents[bool_]
        # Must remain the same length as rectangles
        non_parents = np.hstack([non_parents, np.zeros(len(hor_rect), dtype=bool)])

        if verbose:
            prec, now = now, time.time()
            print('Time elapsed:      ', int(now - prec), 'seconds')
            print('ver_rect length:   ', len(ver_rect))
            print('rectangles length: ', rectangles.shape[0])
            print('hor_rect length:   ', len(hor_rect), '\n')

    rectangles = np.vstack([rectangles, *ver_rect])

    return rectangles


def plot_rectangles(coefs, rectangles):
    fig, axes = plt.subplots(1, 1, figsize=(3, 200))

    axes.imshow(coefs, interpolation='nearest', aspect='auto', cmap=mpl.cm.bwr, vmin=-coef_max, vmax=coef_max)

    p = mpl.collections.PatchCollection([mpl.patches.Rectangle((r[0], r[1]), r[2], r[3]) for r in rectangles])
    p.set_facecolor('none')
    p.set_edgecolor('k')
    axes.add_collection(p)

    plt.show()


def color_rectangles(coefs, rectangles):
    values = np.zeros(rectangles.shape[0])

    for i, r in enumerate(rectangles):
        array_ = coefs[r[1]:r[1] + r[3] + 1, r[0]:r[0] + r[2] + 1]
        extremums = (np.amin(array_), np.amax(array_))

        if -extremums[0] > extremums[1]:
            values[i] = extremums[0]
        else:
            values[i] = extremums[1]

    return values


def slice(level=None, level_frac=None):
    """Get boundaries intersecting a vertical axis.

    Either use the `x == level` or `x == maximum_level * level_frac` axis.
    """
    if (level is None and level_frac is None)
            or (level is not None and level_frac is not None):
        raise ValueError

    if level_frac is not None:
        level = int(coefs.shape[1] * level_frac)

    boundaries = rectangles[(rectangles[:, 0] <= slice_) & (slice_ < rectangles[:, 0]+rectangles[:, 2])][:, 1]
    boundaries.sort()
    boundaries = list(boundaries) + [coefs.shape[0]]

    return boundaries
