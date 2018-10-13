import os
import pathlib
import random
import re

import cv2
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, Iterator, img_to_array
import numpy as np
from util import path


def get_labels(file_names):
    """
    根据图片名称数组获取图片标签数组
    :param file_names: 图片名称数组
    :return:
    """
    labels = []
    for i in file_names:
        label = i.split(".")[-2].split("_")[1:]
        labels.append(list(map(int, label)))
    return labels


def get_label(file_name):
    """
    从图片的名称获取其标签
    :param file_name: 图片名
    :return:
    """
    label = file_name.split(".")[-2].split("_")[1:]
    return label


def get_k_fold_files(val_index, shuffle=True):
    """
    获得指定fold作为cv，其余作为train的图片文件名
    :param val_index: cv
    :param shuffle: 是否打乱
    :return:
    """
    train_names = []
    val_names = []
    with open(path.K_FOLD_FILE, 'r') as f:
        for l in f.readlines():
            k, name = l.split(",")
            val_names.append(name.strip()) if int(k) is val_index else train_names.append(name.strip())

    train_files = []
    val_files = []

    for name in train_names:
        train_files.append(os.path.join(path.ORIGINAL_TRAIN_IMAGES_PATH, name))
    for name in val_names:
        val_files.append(os.path.join(path.ORIGINAL_TRAIN_IMAGES_PATH, name))

    # 不对validation 数据集进行shuffle, 确保所有模型evaluate得出的结果是能够对应的，便于快速ensemble
    if shuffle:
        random.shuffle(train_files)

    return train_files, val_files


def list_image_dir(directory, ext='jpg|jpeg|bmp|png|ppm'):
    """
    列出目录下的所有图片的路径
    :param directory:
    :param ext:
    :return:
    """
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


def list_image_name(directory, ext='jpg|jpeg|bmp|png|ppm'):
    """
    列出目录下的所有图片的名称
    :param directory:
    :param ext:
    :return:
    """
    return [f for root, _, files in os.walk(directory)
            for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f)]


def load_label(directory, number=None):
    """
    导入指定目录的所有图片的标签，不导入图片
    :param directory:
    :return:
    """
    names = list_image_name(directory)
    random.shuffle(names)
    if number is not None:
        names = names[:number]
    labels = []
    for name in names:
        label = name.split(".")[-2].split("_")[1:]
        labels.append(list(map(int, label)))
    return np.array(labels), np.array(names)

