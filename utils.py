from collections import namedtuple

import numpy as np
import skimage
from PIL import Image

import selectivesearch

Rect = namedtuple('Rect', ['x', 'y', 'w', 'h'])


def pil2nparray(pil_img):
    pil_img.load()
    return np.asarray(pil_img, dtype='float32')


def crop_image(sk_img, rect):
    return sk_img[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w, :]


def image_proposal(im_path):
    im = skimage.io.imread(im_path)
    _, regions = selectivesearch.selective_search(
        im, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        if r['size'] < 220:
            continue

        # resize to 227 * 227
        rx, ry, rw, rh = r['rect']
        rec = Rect(rx, ry, rw, rh)
        img_cropped = crop_image(im, rec)

        if rw == 0 or rh == 0:
            continue
        if len(img_cropped) == 0:
            continue
        [a, b, c] = img_cropped.shape
        if a == 0 or b == 0 or c == 0:
            continue

        candidates.add(r['rect'])

        pil_cropped = Image.fromarray(img_cropped, 'RGB')
        pil_resized = pil_cropped.resize((227, 227), Image.ANTIALIAS)
        img_float = pil2nparray(pil_resized)

        images.append(img_float)
        vertices.append(r['rect'])

    return images, vertices
