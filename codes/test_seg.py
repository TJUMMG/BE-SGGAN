import os
import os.path
import glob
import cv2
import numpy as np
import torch
import torchvision.utils

import models.seg_arch as seg_arch
from data.util import modcrop

# options
test_img_folder_name = '/home/dh/dqq/papercode/BE-SGGAN/datasets/OSTest'  # image folder name
# test_img_folder_name = '/home/dh/dqq/papercode/BE-SGGAN/datasets/Sintel'
test_img_folder = test_img_folder_name + '/LBD'  # LBD images
save_seg_path = test_img_folder_name + '/seg'  # semantic segmentaion color results

# make dirs
if not os.path.exists(save_seg_path):
    os.makedirs(save_seg_path)

# load model
seg_model = seg_arch.OutdoorSceneSeg()
model_path = '/home/dh/dqq/papercode/BE-SGGAN/experiments/pretrained_model/segmentation_OST_bic.pth'
seg_model.load_state_dict(torch.load(model_path), strict=True)
seg_model.eval()
seg_model = seg_model.cuda()

# look_up table # RGB
lookup_table = torch.from_numpy(
    np.array([
        [153, 153, 153],  # 0, background
        [0, 255, 255],  # 1, sky
        [109, 158, 235],  # 2, water
        [183, 225, 205],  # 3, grass
        [153, 0, 255],  # 4, mountain
        [17, 85, 204],  # 5, building
        [106, 168, 79],  # 6, plant
        [224, 102, 102],  # 7, animal
        [255, 255, 255],  # 8/255, void
    ])).float()
lookup_table /= 255

print('seg testing...')

idx = 0
for path in glob.glob(test_img_folder + '/*'):
    idx += 1
    basename = os.path.basename(path)
    base = os.path.splitext(basename)[0]
    print(idx, base)
    # read image
    img = cv2.imread(path)
    img = modcrop(img, 8)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

    img[0] -= 103.939
    img[1] -= 116.779
    img[2] -= 123.68
    img = img.unsqueeze(0)
    img = img.cuda()
    with torch.no_grad():
        output = seg_model(img).detach().float().cpu().squeeze_()  # prob, 1x8xHxW

    # byte img
    _, argmax = torch.max(output, 0)
    argmax = argmax.squeeze().byte()

    # color img
    im_h, im_w = argmax.size()
    color = torch.FloatTensor(3, im_h, im_w).fill_(0)  # black
    for i in range(8):
        mask = torch.eq(argmax, i)
        color.select(0, 0).masked_fill_(mask, lookup_table[i][0])  # R
        color.select(0, 1).masked_fill_(mask, lookup_table[i][1])  # G
        color.select(0, 2).masked_fill_(mask, lookup_table[i][2])  # B
    # void
    mask = torch.eq(argmax, 255)
    color.select(0, 0).masked_fill_(mask, lookup_table[8][0])  # R
    color.select(0, 1).masked_fill_(mask, lookup_table[8][1])  # G
    color.select(0, 2).masked_fill_(mask, lookup_table[8][2])  # B
    torchvision.utils.save_image(color, os.path.join(save_seg_path, base+'.png'), padding=0, \
                                 normalize=False)
