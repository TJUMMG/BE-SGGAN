# BE-SGGAN
- This repo only provides testing codes - **pytorch version**. If you use this code, please cite the following publication: J.Liu, Q.Dou, Y.Su, P.Jing, "BE-SGGAN: Texture Realistic Bit-Depth Enhancement by Semantic Guided GAN", to appear in

## Dependencies
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 0.4.0](https://pytorch.org/)
- Python packages:  `pip install numpy opencv-python`

## How to Test
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
- `./metrics/`: calculates PSNR and SSIM.

### Datasets
- There are two sample images in the `./OSTest` folder.

### Experiments
- `./pretrained-model/segmentation_OST_bic`: segmentation model for outdoor scenes.
- `./models/BE-SGGAN_300000_G`: BE-SGGAN model, trained on OST and DIV2K for 4-8 bit BDE task, can be used for 4-16 bit BDE directly.

## Acknowledgement
- Thanks to *X.Wang et al*, who are authors of "Recovering realistic texture in image super-resolution by deep spatial feature transform", published in the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), for referring to their outstanding work.
