# -*- coding: utf-8 -*-
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import selectivesearch


def visualize(img):
    # loading astronaut image
    # img = skimage.data.astronaut()

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 220:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        # if w / h > 1.2 or h / w > 1.2:
        #    continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for i, r in enumerate(candidates):
        x, y, w, h = r

        print(i, x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()


def print_regions(regions):
    for r in regions:
        print(r)


def show_rect_on_image(img, rect):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    rect = mpatches.Rectangle(
        (rect.x, rect.y), rect.w, rect.h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
    plt.show()


def show_image(img, name):
    plt.imshow(img)
    plt.title(name)
    plt.show()
