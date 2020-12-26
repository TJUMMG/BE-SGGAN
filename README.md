# BE-SGGAN
- This repo only provides testing codes - **pytorch version**. If you use this code, please cite the following publication: J.Liu, Q.Dou, J.Zhang, A.Liu, and X.Yang, "BE-SGGAN: Content-Aware Bit-Depth Enhancement by Semantic Guided GAN".

## Dependencies
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 0.4.0](https://pytorch.org/)
- Python packages:  `pip install numpy opencv-python`

## How to Test
- Download pre-trained models from [Baidu Drive](https://pan.baidu.com/s/1fW0HgsetXqTp-xUUNcmCQw)(qu5s) to `./experiments/models/`
- Organize data at `./datasets/`
- Obtain the semantic segmentation maps: `python test_seg.py`.
- Run command: `python test_sggan_4-8.py` to recover 8-bit images from 4-bit versions.
- Run command: `python test_sggan_4-16.py` to recover 16-bit images from 4-bit versions.

## Table of Contents
- [Codes]
- [Datasets]
- [Experiments]

### Codes
- `./test_***.py`: the entry point for testing.
- `./models/`: defines the architecture of models.
- `./data/`: image processing
- `./scripts/`: generates LBD images from HBD

### Datasets
- There are two sample images in the `./OSTest` folder.

### Experiments
- `./models/segmentation_OST_bic`: segmentation model for outdoor scenes.
- `./models/BE-SGGAN_300000_G`: BE-SGGAN model, trained on OST and DIV2K for 4-8 bit BDE task, can be used for 4-16 bit BDE directly.

##
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

## Acknowledgement
- Thanks to *X.Wang et al*, who are authors of "Recovering realistic texture in image super-resolution by deep spatial feature transform", published in the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), for referring to their outstanding work.
