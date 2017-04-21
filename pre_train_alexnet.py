from __future__ import division, print_function, absolute_import

import os.path

import numpy as np
import tflearn
import tflearn.datasets.oxflower17 as oxflower17
from PIL import Image
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import utils


def create_alexnet(num_classes):
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
    network = dropout(network, 0.5)
    network = fully_connected(network, num_classes, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


def pre_train():
    if os.path.isfile('models/pre_train.model.index'):
        print("Previous trained model exist.")
        return

    X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
    net = create_alexnet(17)
    model = tflearn.DNN(net, checkpoint_path='ckps/pre_train.ckp',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='tmp/pre_train_logs/')
    if os.path.isfile('models/pre_train.model'):
        model.load('models/pre_train.model')
    model.fit(X, Y, n_epoch=100, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='pre_train')
    # Save the model
    model.save('models/pre_train.model')


def test():
    images = []
    imgs = ['0/image_0001.jpg', '10/image_0801.jpg', '15/image_1201.jpg']
    for im in imgs:
        img = Image.open('17flowers/jpg/' + im)
        img = img.resize((227, 227))
        img = utils.pil2nparray(img)
        images.append(img)
    net = create_alexnet(17)
    model = tflearn.DNN(net)
    model.load('models/pre_train.model')
    preds = model.predict(images)
    results = np.argmax(preds, 1)
    print(results)
