import os
import pathlib

import util

# 工程根目录
ROOT_PATH = os.path.abspath(os.sep.join(util.__file__.split(os.sep)[:-2]))

DATA_PATH = os.path.join(ROOT_PATH, "data")
IMAGES_PATH = os.path.join(DATA_PATH, "images")
ORIGINAL_IMAGES_PATH = os.path.join(IMAGES_PATH, "original")

TRAIN_IMAGES_SUBDIR = "train"
TEST_IMAGES_SUBDIR = "test"

ORIGINAL_TRAIN_IMAGES_PATH = os.path.join(ORIGINAL_IMAGES_PATH, TRAIN_IMAGES_SUBDIR)
ORIGINAL_TEST_IMAGES_PATH = os.path.join(ORIGINAL_IMAGES_PATH, TEST_IMAGES_SUBDIR)

TXT_PATH = os.path.join(DATA_PATH, "txt")
TRAIN_DATA_TXT = os.path.join(TXT_PATH, "train-image.txt")
TEST_DATA_TXT = os.path.join(TXT_PATH, "test-image.txt")
TEST_RESULT_TXT = os.path.join(TXT_PATH, "test-label.txt")
K_FOLD_FILE = os.path.join(TXT_PATH, "train-image-k-fold.txt")


def image_path_init():
    pathlib.Path(ORIGINAL_TRAIN_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(ORIGINAL_TEST_IMAGES_PATH).mkdir(parents=True, exist_ok=True)


# 初始化必要的工程目录
image_path_init()
