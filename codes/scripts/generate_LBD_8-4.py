import os, sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.util import imredeep8_np


def generate_mod_LBD_ZP():
    # set parameters
    scale = 4  # 8-4

    # set data dir
    sourcedir = '/home/dh/dqq/papercode/BE-SGGAN/datasets/OSTest/GT'
    savedir = '/home/dh/dqq/papercode/BE-SGGAN/datasets/OSTest/LBD'

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    filepaths = [f for f in os.listdir(sourcedir) if f.endswith('.png')]
    num_files = len(filepaths)

    # prepare data with augementation
    for i in range(num_files):
        filename = filepaths[i]
        print('No.{} -- Processing {}'.format(i, filename))

        # read GT image
        image_GT = cv2.imread(os.path.join(sourcedir, filename))

        # LBD
        image_LBD = imredeep8_np(image_GT, scale)
        cv2.imwrite(os.path.join(savedir, filename), image_LBD)


if __name__ == "__main__":
    generate_mod_LBD_ZP()
