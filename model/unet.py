# !/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


__author__ = "Liu Lihao"


class UNetConvBlock1_1(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3):
        super(UNetConvBlock1_1, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class UNetConvBlock1_2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock1_2, self).__init__()
        self.conv2 = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)
        self.conv3 = nn.Conv2d(out_size, out_size, 1, stride=2, groups=out_size, bias=False)
        # self.conv3.weight.data.fill_(1)

    def forward(self, x):
        out = self.activation(x)
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        out = self.conv3(out)
        return out


class UNetConvBlock1_2_2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock1_2_2, self).__init__()
        self.conv2 = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(x)
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock2, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)
        self.conv3 = nn.Conv2d(out_size, out_size, 1, stride=2, groups=out_size, bias=False)
        # self.conv3.weight.data.fill_(1)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        out = self.conv3(out)
        return out


class UNetConvBlock2_2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock2_2, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock3, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)
        self.conv4 = nn.Conv2d(out_size, out_size, 1, stride=2, groups=out_size, bias=False)
        # self.conv4.weight.data.fill_(1)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.batchnorm(out)
        out = self.conv4(out)
        return out


class UNetConvBlock3_2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock3_2, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock4(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock4, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock5(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock5, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size, padding=2, dilation=2)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock6(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock6, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size, padding=2, dilation=2)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock7(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock7, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock8(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetConvBlock8, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, dilation=1)
        self.bridge = nn.Conv2d(256, 256, kernel_size, padding=1)
        # self.bridge.weight.data.normal_(0, 0.01)
        # self.bridge.bias.data.fill_(1)
        self.conv = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    # def center_crop(self, layer, target_size):
    #     batch_size, n_channels, layer_width, layer_height = layer.size()
    #     xy1 = (layer_width - target_size) // 2
    #     return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]
    def forward(self, x, bridge):
        up = self.up(x)
        out = self.activation(self.bridge(bridge) + up)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock9(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetConvBlock9, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, dilation=1)
        # self.up.weight.data.normal_(0, 0.01)
        # self.up.bias.data.fill_(1)
        self.bridge = nn.Conv2d(128, 128, kernel_size, padding=1)
        # self.bridge.weight.data.normal_(0, 0.01)
        # self.bridge.bias.data.fill_(1)
        self.conv = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        # self.conv.weight.data.normal_(0, 0.01)
        # self.conv.bias.data.fill_(1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x, bridge):
        up = self.up(x)
        out = self.activation(self.bridge(bridge) + up)
        out = self.activation(self.conv(out))
        out = self.batchnorm(out)

        return out


class UNetConvBlock10(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetConvBlock10, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, dilation=1)
        # self.up.weight.data.normal_(0, 0.01)
        # self.up.bias.data.fill_(1)
        self.bridge = nn.Conv2d(64, 128, kernel_size, padding=1)
        # self.bridge.weight.data.normal_(0, 0.01)
        # self.bridge.bias.data.fill_(1)
        self.conv = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        # self.conv.weight.data.normal_(0, 0.01)
        # self.conv.bias.data.fill_(1)
        self.activation = activation
        self.activation2 = nn.LeakyReLU(negative_slope=0.02)

    def forward(self, x, bridge):
        up = self.up(x)
        out = self.activation(self.bridge(bridge) + up)
        out = self.activation2(self.conv(out))
        return out


class prediction(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=1, activation=F.tanh, space_dropout=False):
        super(prediction, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, dilation=1)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))

        out = out * 100

        return out


class convrelu(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=1, activation=F.relu, space_dropout=False):
        super(convrelu, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=0)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        return out


class global_network(nn.Module):
    def __init__(self, image_size):
        super(global_network, self).__init__()
        self.oneD = convrelu(316, 512)
        self.twoD = convrelu(512, 512)
        self.threeD = convrelu(512, 512)
        self.fourD = convrelu(512, 512)
        self.image_size = image_size

    def forward(self, x):  # 4 conv+relu layers with 1 x 1 kernel size with 512 depth,
        tmp = self.oneD(x)  # made into h/8 x w/8 x 512 # input: 1 x 1 x 313+3 dimension tensor
        tmp = self.twoD(tmp)
        tmp = self.threeD(tmp)
        out = self.fourD(tmp)  # batch x 1 x 1 x 512

        out = out.repeat(1, 1, int(self.image_size / 8), int(self.image_size / 8))
        return out


class local_network(nn.Module):
    def __init__(self, in_size, out_size, imsize):
        super(local_network, self).__init__()
        self.imsize = imsize
        self.conv = nn.Conv2d(in_size, out_size, 3, padding=1)

    def forward(self, ab_input):
        out = self.conv(ab_input)  # depth 64 red feed for the network
        return out


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.convlayer1_1 = UNetConvBlock1_1(1, 64)
        self.convlayer1_2 = UNetConvBlock1_2(64, 64)
        self.convlayer1_2_2 = UNetConvBlock1_2_2(64, 64)
        self.convlayer2 = UNetConvBlock2(64, 128)
        self.convlayer2_2 = UNetConvBlock2_2(64, 128)
        self.convlayer3 = UNetConvBlock3(128, 256)
        self.convlayer3_2 = UNetConvBlock3_2(128, 256)
        self.convlayer4 = UNetConvBlock4(256, 512)
        self.convlayer5 = UNetConvBlock5(512, 512)  # Dilated Convolution
        self.convlayer6 = UNetConvBlock6(512, 512)  # Dilated Convolution
        self.convlayer7 = UNetConvBlock7(512, 512)
        self.convlayer8 = UNetConvBlock8(512, 256)
        self.convlayer9 = UNetConvBlock9(256, 128)
        self.convlayer10 = UNetConvBlock10(128, 128)

        self.prediction = prediction(128, 2)

        # self.last = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        layer1_1 = self.convlayer1_1(x)

        layer1_2 = self.convlayer1_2(layer1_1)
        layer1_2_2 = self.convlayer1_2_2(layer1_1)
        layer2 = self.convlayer2(layer1_2)
        layer2_2 = self.convlayer2_2(layer1_2)
        layer3 = self.convlayer3(layer2)
        layer3_2 = self.convlayer3_2(layer2)
        layer4 = self.convlayer4(layer3)

        layer5 = self.convlayer5(layer4)
        layer6 = self.convlayer6(layer5)
        layer7 = self.convlayer7(layer6)
        layer8 = self.convlayer8(layer7, layer3_2)
        layer9 = self.convlayer9(layer8, layer2_2)
        layer10 = self.convlayer10(layer9, layer1_2_2)

        prediction = self.prediction(layer10)

        return prediction


def main():
    a = np.random.randn(2, 1, 224, 224)
    b = np.random.randn(2, 2, 224, 224)
    tensor_5d_a = Variable(torch.from_numpy(a).float().cuda())
    tensor_5d_b = Variable(torch.from_numpy(b).float().cuda())

    model = UNet().cuda()
    score = model(tensor_5d_a)

    print model
    print "score", score.size()
    print "target", tensor_5d_b.size()


if __name__ == '__main__':
    main()