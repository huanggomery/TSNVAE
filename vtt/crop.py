# 裁剪原始的视觉图像，并将原始数据拷贝到新的地方

import os
from PIL import Image
import shutil

ORIGIN_PATH = "data/origin"
DEST_PATH = "data/cropped"
crop_area = [210, 240, 450, 480]
S = 8

def get_subdirectories(directory):
    subdirectories = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            subdirectories.append(dir)
    return subdirectories

def mkdir(name):
    if not os.path.exists(name):
        os.mkdir(name)

mkdir(DEST_PATH)

trajs = get_subdirectories(ORIGIN_PATH)
for dir in trajs:
    mkdir(os.path.join(DEST_PATH, dir))
    for i in range(S):
        I_src = os.path.join(ORIGIN_PATH, dir, "{}.jpg".format(i))
        T_left_src = os.path.join(ORIGIN_PATH, dir, "tac_left_{}.jpg".format(i))
        T_right_src = os.path.join(ORIGIN_PATH, dir, "tac_right_{}.jpg".format(i))
        pos_src = os.path.join(ORIGIN_PATH, dir, "label.npy")

        I_dest = os.path.join(DEST_PATH, dir, "{}.jpg".format(i))
        T_left_dest = os.path.join(DEST_PATH, dir, "tac_left_{}.jpg".format(i))
        T_right_dest = os.path.join(DEST_PATH, dir, "tac_right_{}.jpg".format(i))
        pos_dest = os.path.join(DEST_PATH, dir, "label.npy")

        I = Image.open(I_src)
        I_cropped = I.crop(crop_area)
        I_cropped.save(I_dest)

        shutil.copy(T_left_src, T_left_dest)
        shutil.copy(T_right_src, T_right_dest)
        shutil.copy(pos_src, pos_dest)
