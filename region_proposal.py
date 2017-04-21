from __future__ import division, print_function, absolute_import

import os.path
import pickle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import skimage
from PIL import Image

import selectivesearch
import utils


def iou(rect1, rect2):
    """
    Compute the Intersection Over Union given two rectangles
    :param rect1: Tuple of (x, y, w, h)
    :param rect2: Tuple of (x, y, w, h)
    :return: IOU of float type.
    """

    # If they intersect at x direction
    min_x = min(rect1.x, rect2.x)
    max_x = max(rect1.x + rect1.w, rect2.x + rect2.w)
    is_x = rect1.w + rect2.w - (max_x - min_x)

    # If they intersect at y direction
    min_y = min(rect1.y, rect2.y)
    max_y = max(rect1.y + rect1.h, rect2.y + rect2.h)
    is_y = rect1.h + rect2.h - (max_y - min_y)

    if is_x <= 0 or is_y <= 0:
        return 0.0
    else:
        is_area = is_x * is_y
        area1 = rect1.w * rect1.h
        area2 = rect2.w * rect2.h
        return is_area / (area1 + area2 - is_area)


def propose_and_write(train_list, num_clss, threshold=0.5, svm=False, save_path='data/dataset.pkl'):
    if not os.path.exists('data'):
        os.mkdir('data')

    with open(train_list, 'r') as f:
        labels = []
        images = []
        for line in f:
            tmp = line.strip().split(' ')
            img_addr, img_label, img_rect = tmp[:3]
            print(img_addr, img_label, img_rect)

            img_label = int(img_label)

            img = skimage.io.imread(img_addr)
            _, regions = selectivesearch.selective_search(
                img, scale=500, sigma=0.9, min_size=10)

            candicates = set()
            for r in regions:
                if r['rect'] in candicates:
                    continue
                if r['size'] < 220:
                    continue

                # crop images
                x, y, w, h = r['rect']
                rect = utils.Rect(x, y, w, h)
                img_cropped = utils.crop_image(img, rect)

                # filter cropped image
                if w == 0 or h == 0:
                    continue
                if len(img_cropped) == 0:
                    continue
                a, b, c = img_cropped.shape
                if a == 0 or b == 0 or c == 0:
                    continue

                candicates.add(r['rect'])

                pil_cropped = Image.fromarray(img_cropped, 'RGB')
                pil_resized = pil_cropped.resize((227, 227), Image.ANTIALIAS)

                img_float = utils.pil2nparray(pil_resized)
                images.append(img_float)

                # Compute iou
                gt_list = img_rect.split(',')
                x2, y2, w2, h2 = map(int, gt_list)
                gt_rect = utils.Rect(x2, y2, w2, h2)
                iou_val = iou(gt_rect, rect)

                # debug
                if False:
                    if iou_val < threshold:
                        label = 0
                    else:
                        label = img_label

                    pil_img = Image.fromarray(img, 'RGB')
                    ax = plt.subplot(111)
                    ax.imshow(pil_img)
                    r_gt = patches.Rectangle((x2, y2), w2, h2, linewidth=1, edgecolor='b', facecolor='none')
                    r_cu = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(r_gt)
                    ax.add_patch(r_cu)
                    plt.title('{} {}'.format(label, iou_val))
                    plt.show()

                if svm:
                    if iou_val < threshold:
                        labels.append(0)
                    else:
                        labels.append(img_label)
                else:
                    label = np.zeros(num_clss + 1)
                    if iou_val < threshold:
                        label[0] = 1
                    else:
                        label[img_label] = 1
                    labels.append(label)
        # save dataset
        pickle.dump((images, labels), open(save_path, 'wb'))


def load_from_pkl(dataset_file):
    X, Y = pickle.load(open(dataset_file, 'rb'))
    return X, Y
