'''
architecture for sft
'''
import torch.nn as nn
import torch.nn.functional as F


class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_shared_conv0 = nn.Conv2d(3, 32, 1)
        self.SFT_scale_conv0 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        shared = F.leaky_relu(self.SFT_shared_conv0(x[1]), 0.1, inplace=True)
        scale = self.SFT_scale_conv0(shared)
        shift = self.SFT_shift_conv0(shared)
        return x[0] * (scale + 1) + shift


class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions


class ResBlock_GAN(nn.Module):
    def __init__(self):
        super(ResBlock_GAN, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        # x: fea
        fea = F.relu(self.conv0(x[0]), inplace=True)
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions


class SFT_Net(nn.Module):
    def __init__(self):
        super(SFT_Net, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)

        sft_branch = []
        for i in range(4):
            sft_branch.append(ResBlock_SFT())
            sft_branch.append(ResBlock_GAN())
            sft_branch.append(ResBlock_GAN())
            sft_branch.append(ResBlock_GAN())
        sft_branch.append(SFTLayer())
        # sft_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.sft_branch = nn.Sequential(*sft_branch)

        self.HR_branch = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, x):
        # x[0]: img; x[1]: color
        cond = x[1]
        fea = self.conv0(x[0])
        sft = self.sft_branch((fea, cond))
        res = self.HR_branch(sft)
        out = x[0] + res
        return out


# Auxiliary Classifier Discriminator
class ACD_VGG_BN_96(nn.Module):
    def __init__(self):
        super(ACD_VGG_BN_96, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),
        )

        # gan
        self.gan = nn.Sequential(
            nn.Linear(512*6*6, 100),
            nn.LeakyReLU(0.1, True),
            nn.Linear(100, 1)
        )

        self.cls = nn.Sequential(
            nn.Linear(512*6*6, 100),
            nn.LeakyReLU(0.1, True),
            nn.Linear(100, 8)
        )

    def forward(self, x):
        fea = self.feature(x)
        fea = fea.view(fea.size(0), -1)
        gan = self.gan(fea)
        cls = self.cls(fea)
        return [gan, cls]

