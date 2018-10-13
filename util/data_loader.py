import os
import re
import time
import random
import pathlib
import concurrent.futures

import cv2
import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, Iterator, img_to_array

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


def parse_data_line(line: str):
    """
    处理图片官方提供的数据，得到图片编号，url，类型

    :param line: 待处理的数据
    :return: 图片编号、URL、类型
    """

    pieces = line.strip().split(",")
    return pieces[0], pieces[1], "_".join(pieces[2:])


def do_download(lines: list, photo_save_dir: str, photo_save_subdir: str, is_test: bool):
    """
    下载图片

    :param lines: 待下载的数据
    :param photo_save_dir: 存储下载图片的根目录
    :param photo_save_subdir: 存储下载图片的子目录，根目录+子目录构成完整目录
    :param is_test: txt_dir对应的文本是否是测试数据（测试数据和训练数据的格式不同）
    :return:
    """

    print("开始下载图片")
    for n in lines:
        name, url, label = parse_data_line(n)
        while True:
            download_ok = True
            try:
                if is_test:
                    tf.keras.utils.get_file(fname=name + ".jpg", origin=url, cache_dir=photo_save_dir,
                                            cache_subdir=photo_save_subdir)
                else:
                    tf.keras.utils.get_file(fname="_".join([name, label]) + ".jpg", origin=url,
                                            cache_dir=photo_save_dir,
                                            cache_subdir=photo_save_subdir)
            except Exception as e:
                download_ok = False
                print(e)
                print("start retry")
            if download_ok:
                break


def download_images(txt_dir: str, photo_save_dir: str, photo_save_subdir: str, is_test: bool, thread_number: int = 2):
    """ 使用多线程下载图片到制定的目录，线程过多可能导致服务器拒绝，2个线程比较稳定
        如果中途失败直接重试即可，不会重复下载已下载的图片

    :param txt_dir: 记录数据的文本的路径
    :param photo_save_dir: 存储下载图片的根目录
    :param photo_save_subdir: 存储下载图片的子目录，根目录+子目录构成完整目录
    :param is_test: txt_dir对应的文本是否是测试数据（测试数据和训练数据的格式不同）
    :param thread_number: 使用多线程下载的线程数量
    :return:
    """

    pathlib.Path(os.path.join(photo_save_dir, photo_save_subdir)).mkdir(parents=True, exist_ok=True)

    with open(txt_dir) as f:
        lines = []
        for i in range(thread_number):
            lines.append([])
        line_num = 0
        for l in f.readlines():
            lines[line_num % thread_number].append(l)
            line_num += 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_number) as executor:
            futures = []
            start_time = time.time()
            for l in lines:
                futures.append(executor.submit(do_download, l, photo_save_dir, photo_save_subdir, is_test))
            concurrent.futures.wait(futures)
            print("下载所有图片共花费 %f 秒" % (time.time() - start_time))


def download_all_images(thread_number):
    """
    下载训练数据
    :param thread_number: 并发下载数量
    :return:
    """
    download_images(path.TRAIN_DATA_TXT, path.ORIGINAL_IMAGES_PATH, path.TRAIN_IMAGES_SUBDIR, is_test=False,
                    thread_number=thread_number)
    download_images(path.TEST_DATA_TXT, path.ORIGINAL_IMAGES_PATH, path.TEST_IMAGES_SUBDIR, is_test=True,
                    thread_number=thread_number)


def image_repair():
    """
    下载下来的图像部分格式存在小问题，通过CV打开再保存即可修复。
    :return:
    """
    names = list_image_dir(path.ORIGINAL_TRAIN_IMAGES_PATH)
    names += list_image_dir(path.ORIGINAL_TEST_IMAGES_PATH)
    for name in names:
        img = cv2.imread(name)
        cv2.imwrite(name, img)


if __name__ == '__main__':
    download_all_images(16)
    image_repair()