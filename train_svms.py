from __future__ import division, print_function, absolute_import

import os
import os.path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import skimage
import tflearn
from PIL import Image
from sklearn import svm
from sklearn.externals import joblib
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import region_proposal as pr
import selectivesearch
import utils


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
        rec = utils.Rect(rx, ry, rw, rh)
        img_cropped = utils.crop_image(im, rec)

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
        img_float = utils.pil2nparray(pil_resized)

        images.append(img_float)
        vertices.append(r['rect'])

    return images, vertices


def generate_single_svm_train_data(one_class_train_file):
    train_file = one_class_train_file
    save_path = one_class_train_file.replace('txt', 'pkl')

    if not os.path.isfile(save_path):
        print("Preparing svm dataset..." + save_path)
        pr.propose_and_write(train_file, 2, threshold=0.5, svm=True, save_path=save_path)
    print("Restoring svm dataset..." + save_path)
    images, labels = pr.load_from_pkl(save_path)
    return images, labels


# Use a already trained alexnet with the last layer redesigned
def create_alexnet():
    network = input_data(shape=[None, 227, 227, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


def train_svms():
    if not os.path.isfile('models/fine_tune.model.index'):
        print('models/fine_tune.model doesn\'t exist.')
        return

    net = create_alexnet()
    model = tflearn.DNN(net)
    model.load('models/fine_tune.model')

    train_file_dir = 'svm_train/'
    flist = os.listdir(train_file_dir)
    svms = []
    for train_file in flist:
        if "pkl" in train_file:
            continue
        X, Y = generate_single_svm_train_data(train_file_dir + train_file)
        train_features = []
        for i in X:
            feats = model.predict([i])
            train_features.append(feats[0])
        print("feature dimension of fitting: {}".format(np.shape(train_features)))
        clf = svm.LinearSVC()
        clf.fit(train_features, Y)
        svms.append(clf)
    joblib.dump(svms, 'models/train_svm.model')


def test():
    if not os.path.isfile('models/fine_tune.model.index'):
        print("models/fine_tune_model doesn't exist.")
        return
    if not os.path.isfile('models/train_svm.model'):
        print('models/train_svm.model doesn\'s exist.')
        return

    img_path = '2flowers/jpg/0/image_0561.jpg'
    imgs, verts = image_proposal(img_path)

    net = create_alexnet()
    model = tflearn.DNN(net)
    model.load('models/fine_tune.model')
    features = model.predict(imgs)
    print("feature dimension of testing: {}".format(np.shape(features)))

    svms = joblib.load('models/train_svm.model')
    results_rect = []
    results_label = []
    count = 0
    for f in features:
        for s in svms:
            pred = s.predict(f)
            print("pred:", pred)
            if pred[0] != 0:
                results_rect.append(verts[count])
                results_label.append(pred[0])
        count += 1

    print("result:")
    print(results_rect)
    print("result label:")
    print(results_label)

    img = skimage.io.imread(img_path)
    _, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in results_rect:
        rect = patches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()
