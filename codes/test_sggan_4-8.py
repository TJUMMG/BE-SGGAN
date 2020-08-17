import os.path
import glob
import cv2
import numpy as np
import torch
from data.util import modcrop, tensor2img
import models.sggan_arch as sggan

model_path = '/home/dh/dqq/papercode/BE-SGGAN/experiments/models/BE-SGGAN_300000_G.pth'  # pytorch training
test_img_folder_name = 'OSTest'  # image folder name
test_img_folder = '/home/dh/dqq/papercode/BE-SGGAN/datasets/' + test_img_folder_name + '/LBD'  # LBD images
test_prob_path = '/home/dh/dqq/papercode/BE-SGGAN/datasets/' + test_img_folder_name + '/seg'  # semantic segmentaion maps
save_results_path = '/home/dh/dqq/papercode/BE-SGGAN/datasets/' + test_img_folder_name + '/results'  # HBD results

# make dirs
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)

model = sggan.SFT_Net()
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.cuda()

print('sggan testing...')

idx = 0
for path in glob.glob(test_img_folder + '/*'):
    idx += 1
    basename = os.path.basename(path)
    base = os.path.splitext(basename)[0]
    print(idx, base)
    # read image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = modcrop(img, 8)
    img = img * 1.0 / 255
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img_LBD = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()

    img_LBD = img_LBD.unsqueeze(0)
    img_LBD = img_LBD.cuda()

    # read seg
    seg = cv2.imread(os.path.join(test_prob_path, base + '.png'), cv2.IMREAD_UNCHANGED)
    seg = seg * 1.0 / 255
    seg = torch.from_numpy(np.transpose(seg, (2, 0, 1))).float()

    seg = seg.unsqueeze(0)
    seg = seg.cuda()

    with torch.no_grad():
        output = model((img_LBD, seg)).data
    output = tensor2img(output.squeeze(), out_type=np.uint8)
    cv2.imwrite(os.path.join(save_results_path, base + '_rlt.png'), output)

