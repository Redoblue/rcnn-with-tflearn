from __future__ import division, print_function, absolute_import

import os.path

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import region_proposal as pr


def create_alexnet(num_classes, restore=False):
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
    network = fully_connected(network, num_classes, activation='softmax', restore=restore)
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


def fine_tune():
    print("Starting fine tuning...")
    if not os.path.isfile('data/dataset.pkl'):
        print("Preparing Data...")
        pr.propose_and_write('refine_list.txt', 2)

    print("Loading Data...")
    X, Y = pr.load_from_pkl('data/dataset.pkl')
    print("Loading Done.")

    # whether to restore the last layer
    restore = False
    if os.path.isfile('models/fine_tune.model.index'):
        restore = True
        print("Continue training...")

    net = create_alexnet(3, restore)
    model = tflearn.DNN(net, checkpoint_path='ckps/fine_tune.ckp',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='tmp/fine_tune_logs/')

    if os.path.isfile('models/fine_tune.model.index'):
        print("Loading the fine tuned model")
        model.load('models/fine_tune.model.index')
    elif os.path.isfile('models/pre_train.model'):
        print("Loading the alexnet")
        model.load('models/pre_train.model')
    else:
        print("No file to load, error")
        return False
    model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='fine_tune')  # epoch = 1000
    # Save the model
    model.save('models/fine_tune.model')
