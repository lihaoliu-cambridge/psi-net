# !/usr/bin/env python
# coding=utf-8
from torchvision import models
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import math

__author__ = "Liu Lihao"


def get_upsampling_weight(m):
    factor = (m.kernel_size[0] + 1) // 2
    if m.kernel_size[0] % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:m.kernel_size[0], :m.kernel_size[1]]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((m.in_channels, m.out_channels, m.kernel_size[0], m.kernel_size[1]), dtype=np.float64)
    weight[list(range(m.in_channels)), list(range(m.out_channels)), :, :] = filt

    m.weight.data.copy_(torch.from_numpy(weight).float())


def initialize_weights(layer_modules):
    for m in layer_modules:
        if isinstance(m, nn.Conv2d):
            num = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / num))
            # m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            num = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / num))
            # m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            get_upsampling_weight(m)


def test_initialization(layer_modules):
    for m in layer_modules:
        if isinstance(m, nn.Conv3d):
            print "w", m.weight.data
            print "b", m.bias.data


class ColorNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ColorNet, self).__init__()

        # Low-Level
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        # Mid-Level
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        # Colorization
        self.conv9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(128)
        self.upsampling1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv10 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(64)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.upsampling2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv12 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(32)
        self.conv13 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm2d(2)
        self.upsampling3 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.net = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,

            self.conv2,
            self.bn2,
            self.relu,

            self.conv3,
            self.bn3,
            self.relu,

            self.conv4,
            self.bn4,
            self.relu,

            self.conv5,
            self.bn5,
            self.relu,

            self.conv6,
            self.bn6,
            self.relu,

            self.conv7,
            self.bn7,
            self.relu,

            self.conv8,
            self.bn8,
            self.relu,

            self.conv9,
            self.bn9,
            self.relu,

            self.upsampling1,

            self.conv10,
            self.bn10,
            self.relu,

            self.conv11,
            self.bn11,
            self.relu,

            self.upsampling2,

            self.conv12,
            self.bn12,
            self.relu,

            self.conv13,
            self.bn13,

            self.upsampling3,

            self.tanh
        )

        initialize_weights(self.modules())

    def forward(self, x):
        return self.net(x) * 100


def main():
    a = np.random.randn(2, 1, 224, 224)
    b = np.random.randn(2, 2, 224, 224)
    tensor_5d_a = Variable(torch.from_numpy(a).float().cuda())
    tensor_5d_b = Variable(torch.from_numpy(b).float().cuda())

    model = ColorNet().cuda()
    score = model(tensor_5d_a)

    print model
    print "score", score
    print "target", tensor_5d_b


if __name__ == '__main__':
    main()
