import os

from fine_tune_alexnet import fine_tune
from pre_train_alexnet import pre_train
from train_svms import train_svms, test

def make_dirs():
    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
    if not os.path.isdir('ckps'):
        os.mkdir('ckps')

def run_train():
    make_dirs()

    if not os.path.isfile('models/pre_train.model.index'):
        print("Starting pre training...")
        pre_train()
        print("Pre training completed.")
    if not os.path.isfile('models/fine_tune.model.index'):
        print("Starting pre training...")
        fine_tune()
        print("Pre training completed.")
    if not os.path.isfile('models/train_svm.model'):
        print("Starting fitting svms...")
        train_svms()
        print("fitting svms completed.")


def run_test():
    print("Starting testing...")
    test()


if __name__ == '__main__':
    run_test()
