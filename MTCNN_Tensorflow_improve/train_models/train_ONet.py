#coding:utf-8


"""
Run this script to train ONet
"""

from mtcnn_model import O_Net
from train import train


def train_ONet(base_dir, prefix, end_epoch, display, lr):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    net_factory = O_Net
    train(net_factory, prefix, end_epoch, base_dir, display=display, base_lr=lr)

if __name__ == '__main__':
    base_dir = '../prepare_data/imglists/ONet'

    model_name = 'MTCNN'
    model_path = '../data/%s_model/ONet_landmark/ONet' % model_name
    prefix = model_path
    end_epoch = 36
    display = 1000
    lr = 0.01
    train_ONet(base_dir, prefix, end_epoch, display, lr)
