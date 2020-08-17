import os
import math
import pickle
import random
import numpy as np
import lmdb
import torch
import cv2
import logging


####################
# image processing
# process on numpy image
####################

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


####################
# Functions
####################

def imredeep(img, scale):
    in_H, in_W, in_C = img.size()
    scale = scale

    out = torch.FloatTensor(in_H, in_W, in_C)
    for i in range(in_H):
        for j in range(in_W):
            for k in range(in_C):
                tmp = bin(int(img[i, j, k]))
                tmp_quan = tmp[:-scale] + '0' * scale
                out[i, j, k] = int(tmp_quan, 2)

    return out


def imredeep8_np(img, scale):
    img = torch.from_numpy(img)
    in_H, in_W, in_C = img.size()
    scale = scale

    out = torch.FloatTensor(in_H, in_W, in_C)
    for i in range(in_H):
        for j in range(in_W):
            for k in range(in_C):
                tmp = bin(int(img[i, j, k]))
                tmp_quan = tmp[:-scale] + '0' * scale
                out[i, j, k] = int(tmp_quan, 2)

    return np.array(out, dtype=np.uint8)


def imredeep16_np(img, scale):
    img = torch.from_numpy(img / 1.)
    in_H, in_W, in_C = img.size()
    scale = scale

    out = torch.FloatTensor(in_H, in_W, in_C)
    for i in range(in_H):
        for j in range(in_W):
            for k in range(in_C):
                tmp = bin(int(img[i, j, k]))
                tmp_quan = tmp[:-scale] + '0' * scale
                out[i, j, k] = int(tmp_quan, 2)

    return np.array(out, dtype=np.uint16)


####################
# image convert
####################

def tensor2img(tensor, out_type, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    out_type = out_type
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    if out_type == np.uint16:
        img_np = (img_np * 65535.0).round()
    return img_np.astype(out_type)

