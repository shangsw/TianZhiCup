import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.functional import interpolate as interpolate
from .carafe import CARAFE
import math
from torch.nn.modules.batchnorm import _BatchNorm


def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()

    return x1, x2

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class Conv2dBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch, eps=1e-3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# after Concat -> BN, you also can use Dropout like SS_nbt_module may be make a good result!
class DownsamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel - in_channel, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output


class SS_nbt_module(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        oup_inc = chann // 2

        # dw
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                     dilation=(dilated, 1))

        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                     dilation=(1, dilated))

        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        # dw
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                     dilation=(dilated, 1))

        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                     dilation=(1, dilated))

        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)


    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, input):
        # x1 = input[:,:(input.shape[1]//2),:,:]
        # x2 = input[:,(input.shape[1]//2):,:,:]
        residual = input
        x1, x2 = split(input)

        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1 = self.relu(output1)

        output1 = self.conv3x1_2_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)

        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2 = self.relu(output2)

        output2 = self.conv1x3_2_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)

        if (self.dropout.p != 0):
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)

        out = self._concat(output1, output2)
        out = F.relu(residual + out, inplace=True)
        return channel_shuffle(out, 2)

class SS_nbt_module1(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        oup_inc = chann // 2

        # dw
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3, 3), stride=1, padding=(1 * dilated, 1 * dilated), bias=True,
                                     dilation=(dilated,dilated))

        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (3, 3), stride=1, padding=(1 * dilated, 1 * dilated), bias=True,
                                     dilation=(dilated, dilated))

        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        # dw
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3, 3), stride=1, padding=(1 * dilated,1 * dilated), bias=True,
                                     dilation=(dilated, dilated))

        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (3, 3), stride=1, padding=(1 * dilated, 1 * dilated), bias=True,
                                     dilation=(dilated, dilated))

        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)


    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, input):
        # x1 = input[:,:(input.shape[1]//2),:,:]
        # x2 = input[:,(input.shape[1]//2):,:,:]
        residual = input
        x1, x2 = split(input)

        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1 = self.relu(output1)

        output1 = self.conv3x1_2_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)

        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2 = self.relu(output2)

        output2 = self.conv1x3_2_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)

        if (self.dropout.p != 0):
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)

        out = self._concat(output1, output2)
        out = F.relu(residual + out, inplace=True)
        return channel_shuffle(out, 2)

class SS_nbt_module2(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        oup_inc = chann // 4

        # dw
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                     dilation=(dilated, 1))

        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                     dilation=(1, dilated))

        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        # dw
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                     dilation=(dilated, 1))

        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                     dilation=(1, dilated))

        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)


    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, input):
        # x1 = input[:,:(input.shape[1]//2),:,:]
        # x2 = input[:,(input.shape[1]//2):,:,:]
        residual = input
        x1, x2 = split(input)
        x0, x1 = split(x1)
        x2, x3 = split(x2)

        output0 = self.conv3x1_1_l(x0)
        output0 = self.relu(output0)
        output0 = self.conv1x3_1_l(output0)
        output0 = self.bn1_l(output0)
        output0 = self.relu(output0)

        output1 = self.conv3x1_2_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)

        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2 = self.relu(output2)

        output3 = self.conv1x3_2_r(x3)
        output3 = self.relu(output3)
        output3 = self.conv3x1_2_r(output3)
        output3 = self.bn2_r(output3)

        if (self.dropout.p != 0):
            output0 = self.dropout(output0)
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)
            output3 = self.dropout(output3)

        out1 = self._concat(output0, output1)
        out2 = self._concat(output2, output3)
        out = self._concat(out1, out2)
        out = F.relu(residual + out, inplace=True)
        return channel_shuffle(out, 2)

class SS_nbt_module3(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        oup_inc = chann // 4

        # dw
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                     dilation=(dilated, 1))

        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                     dilation=(1, dilated))

        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        # dw
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                     dilation=(dilated, 1))

        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                     dilation=(1, dilated))

        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)

        self.conv3x3_1 = nn.Conv2d(2*oup_inc, oup_inc//2, (3, 3), stride=1, padding=(1, 1), bias=True)
        self.conv3x3_2 = nn.Conv2d(oup_inc//2, oup_inc*2, (3, 3), stride=1, padding=(1 * dilated,1 * dilated), bias=True,
                                     dilation=(dilated, dilated))
        self.bn = nn.BatchNorm2d(oup_inc*2, eps=1e-03)

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, input):
        # x1 = input[:,:(input.shape[1]//2),:,:]
        # x2 = input[:,(input.shape[1]//2):,:,:]
        residual = input
        x1, x2 = split(input)
        x0, x1 = split(x1)
        x2, x3 = split(x2)
        x3 = self._concat(x0, x3)

        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1 = self.relu(output1)

        output1 = self.conv3x1_2_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)

        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2 = self.relu(output2)

        output2 = self.conv1x3_2_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)

        output3_1 = self.conv3x3_1(x3)
        output3_1 = self.relu(output3_1)
        output3_1 = self.conv3x3_2(output3_1)
        output3_1 = self.bn(output3_1)

        output3_2 = self.conv3x3_1(x3)
        output3_2 = self.relu(output3_2)
        output3_2 = self.conv3x3_2(output3_2)
        output3_2 = self.bn(output3_2)

        output3_3 = self.conv3x3_1(x3)
        output3_3 = self.relu(output3_3)
        output3_3 = self.conv3x3_2(output3_3)
        output3_3 = self.bn(output3_3)

        output3_4 = self.conv3x3_1(x3)
        output3_4 = self.relu(output3_4)
        output3_4 = self.conv3x3_2(output3_4)
        output3_4 = self.bn(output3_4)

        output3_5 = self.conv3x3_1(x3)
        output3_5 = self.relu(output3_5)
        output3_5 = self.conv3x3_2(output3_5)
        output3_5 = self.bn(output3_5)

        output3_6 = self.conv3x3_1(x3)
        output3_6 = self.relu(output3_6)
        output3_6 = self.conv3x3_2(output3_6)
        output3_6 = self.bn(output3_6)

        output3_7 = self.conv3x3_1(x3)
        output3_7 = self.relu(output3_7)
        output3_7 = self.conv3x3_2(output3_7)
        output3_7 = self.bn(output3_7)

        output3_8 = self.conv3x3_1(x3)
        output3_8 = self.relu(output3_8)
        output3_8 = self.conv3x3_2(output3_8)
        output3_8 = self.bn(output3_8)

        output3 = output3_1 + output3_2 + output3_3 + output3_4 + output3_5 + output3_6 + output3_7 + output3_8

        if (self.dropout.p != 0):
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)
            output3 = self.dropout(output3)

        out = self._concat(output1, output2)
        out = self._concat(out, output3)
        out = F.relu(residual + out, inplace=True)
        return channel_shuffle(out, 2)

class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # for 3 channels
        self.initial_block = DownsamplerBlock(3,32)
        # for 1 channel
        # self.initial_block = DownsamplerBlock(1, 32)

        self.layers = nn.ModuleList()

        for x in range(0, 3):
            self.layers.append(SS_nbt_module(32, 0.03, 1))

        self.layers.append(DownsamplerBlock(32, 64))

        for x in range(0, 2):
            self.layers.append(SS_nbt_module(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 1):
            self.layers.append(SS_nbt_module(128, 0.3, 1))
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))

        for x in range(0, 1):
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))
            self.layers.append(SS_nbt_module(128, 0.3, 17))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):

        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output

class Encoder1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # for 3 channels
        # self.initial_block = DownsamplerBlock(3,64)
        # for 1 channel
        self.initial_block = DownsamplerBlock(1, 64)

        self.layers = nn.ModuleList()

        for x in range(0, 3):
            self.layers.append(SS_nbt_module(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):
            self.layers.append(SS_nbt_module(128, 0.03, 1))

        self.layers.append(DownsamplerBlock(128, 256))

        for x in range(0, 1):
            self.layers.append(SS_nbt_module3(256, 0.3, 1))
            self.layers.append(SS_nbt_module3(256, 0.3, 2))
            self.layers.append(SS_nbt_module3(256, 0.3, 5))
            self.layers.append(SS_nbt_module3(256, 0.3, 9))

        for x in range(0, 1):
            self.layers.append(SS_nbt_module3(256, 0.3, 2))
            self.layers.append(SS_nbt_module3(256, 0.3, 5))
            self.layers.append(SS_nbt_module(256, 0.3, 9))
            self.layers.append(SS_nbt_module(256, 0.3, 17))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(256, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):

        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output
class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=True)
        return x


class APN_Module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(APN_Module, self).__init__()
        self.out_ch = out_ch
        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )
        # midddle branch
        self.mid = nn.Sequential(
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )
        self.down1 = Conv2dBnRelu(in_ch, 1, kernel_size=7, stride=2, padding=3)

        self.down2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=2, padding=2)

        self.down3 = nn.Sequential(
            Conv2dBnRelu(1, 1, kernel_size=3, stride=2, padding=1),
            Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1)
        )

        self.conv2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(1, 1, kernel_size=7, stride=1, padding=3)
        # self.mid = nn.Sequential(
        #     Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        # )
        #
        # self.down1 = Conv2dBnRelu(in_ch, 128, kernel_size=7, stride=2, padding=3)
        #
        # self.down2 = Conv2dBnRelu(128, 128, kernel_size=5, stride=2, padding=2)
        #
        # self.down3 = nn.Sequential(
        #     Conv2dBnRelu(128, 128, kernel_size=3, stride=2, padding=1),
        #     Conv2dBnRelu(128, 5, kernel_size=1, stride=1, padding=0)
        # )
        # self.conv2 = Conv2dBnRelu(128, 5, kernel_size=1, stride=1, padding=0)
        # self.conv1 = Conv2dBnRelu(128, 5, kernel_size=1, stride=1, padding=0)

        self.carafe_ = CARAFE(5,scale=64).cuda()
        self.carafe = CARAFE(1).cuda()

        # self.emau = EMAU(1, 64)
    def forward(self, x):
        # print(x.size(),'33333333333333333')
        h = x.size()[2]
        w = x.size()[3]

        b1 = self.branch1(x)
        # print(b1.size(),'!!!!!!!!!!') #(batchsize,nclass,1,1)
        # b1 = Interpolate(size=(h, w), mode="bilinear")(b1)
        # b1 = interpolate(b1, size=(h, w), mode="bilinear", align_corners=True)
        # print(b1.size(),'!!!!!!!!!!') #(batchsize,nclass,64,64)
        # for i in range(6):
        #     b1 = CARAFE(c=b1.size(1)).cuda()(b1)    #carafe
        b1 = self.carafe_(b1) #carafe1
        # b1 = DUpsampling(b1.size(1),scale=64).cuda()(b1)


        mid = self.mid(x)
        # print(x.shape,'000000000000000')

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        ####jiazhewan2#######
        # x4 = self.down4(x3)
        # x4 = interpolate(x4, size=(h // 8, w // 8), mode="bilinear", align_corners=True)
        # x3 = x3 +x4
        #####################

        # x3 = Interpolate(size=(h // 4, w // 4), mode="bilinear")(x3)
        # print(x3.size(),'@@@@@@@@@@@') #(batchsize,1,8,8)
        # x3 = interpolate(x3, size=(h // 4, w // 4), mode="bilinear", align_corners=True)
        # print(x3.size(),'@@@@@@@@@@@') #(batchsize,1,16,16)
        x3 = self.carafe(x3)

        x2 = self.conv2(x2)
        x = x2 + x3
        # print(x.size(),'############') #(batchsize,1,16,16)

        ###添加JPU模块part1
        # x3 = interpolate(x3, size=(h // 2, w // 2), mode="bilinear", align_corners=True)
        # x2 = interpolate(x2, size=(h // 2, w // 2), mode="bilinear", align_corners=True)
        ######################

        # x = Interpolate(size=(h // 2, w // 2), mode="bilinear")(x)
        # x = interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=True)
        # print(x.size(),'############') ##(batchsize,1,32,32)
        x = self.carafe(x)
        x1 = self.conv1(x1)

        ###添加JPU模块part2
        # another_x = torch.cat([x1,x2,x3],dim=1)
        # print(another_x.size()) #(8,3,32,32)
        # another_x = torch.cat([self.dilation1(another_x),self.dilation2(another_x),self.dilation3(another_x),self.dilation4(another_x)],dim=1)
        #################
        x = x + x1
        # print(x.size(),'$$$$$$$$$$$$$$') ##(batchsize,1,32,32)
        # x = self.emau(x)
        ###添加JPU模块part3
        # x = torch.cat([x, another_x],dim=1)
        # x = nn.Conv2d(in_channels=x.size(1), out_channels=self.out_ch, kernel_size=1, stride=1, padding=1, bias=True).cuda()(x)
        ####################

        # x = Interpolate(size=(h, w), mode="bilinear")(x)
        # x = interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        # print(x.size(),'$$$$$$$$$$$$$$') ##(batchsize,1,64,64)
        x = self.carafe(x)

        # print(x.shape,mid.shape,b1.shape,'5555555555555555555')
        x = torch.mul(x, mid)
        # print(x.shape,'66666666666')
        ###添加JPU模块part4
        # x = torch.cat([x,mid],dim=1)
        # x = nn.Conv2d(in_channels=x.size(1), out_channels=self.out_ch, kernel_size=1, stride=1, padding=0, bias=True).cuda()(x)
        ###########################
        x = x + b1
        # print(x.shape,'7777777777777777777')

        return x


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.apn = APN_Module(in_ch=256, out_ch=5)   #original\s1\s11 inch=128,s2\s3 inch=256
        # self.upsample = Interpolate(size=(512, 1024), mode="bilinear")
        # self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True)
        # self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        # self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = self.apn(input)
        # print(input.shape,output.shape,'44444444444444444')#(batchsize,128,64,64),(batchsize,5,64,64)
        out = interpolate(output, size=(512, 512), mode="bilinear", align_corners=True)
        # out = CARAFE(c=output.size(1),scale=8).cuda()(output)
        # out = DUpsampling(output.size(1),scale=8).cuda()(output)

        # out = self.upsample(output)
        # print(out.shape)
        return out


#########JPU相关########
class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


# LEDNet
class Net(nn.Module):
    def __init__(self, num_classes, encoder=None):
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder1(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)


    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)

            # #emu module
            # output = self.emau(output)


            # print(input.shape,output.shape,self.decoder.forward(output).shape,'222222222222222222')
            return self.decoder.forward(output)

            ##############new module###########
            # output = self.decoder.forward(output)
            # return Original(output.size(1)+input.size(1),output.size(1)).cuda()(input,output)
            ###################################


# class EMAU(nn.Module):
#     '''The Expectation-Maximization Attention Unit (EMAU).
#     Arguments:
#         c (int): The input and output channel number.
#         k (int): The number of the bases.
#         stage_num (int): The iteration number for EM.
#     '''
#
#     def __init__(self, c, k, stage_num=3):
#         super(EMAU, self).__init__()
#         self.stage_num = stage_num
#         norm_layer = nn.BatchNorm2d
#         mu = torch.Tensor(1, c, k)
#         mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
#         mu = self._l2norm(mu, dim=1)
#         self.register_buffer('mu', mu)
#
#         self.conv1 = nn.Conv2d(c, c, 1)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(c, c, 1, bias=False),
#             norm_layer(c))
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, _BatchNorm):
#                 m.weight.data.fill_(1)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def forward(self, x):
#         idn = x
#         # The first 1x1 conv
#         x = self.conv1(x)
#
#         # The EM Attention
#         b, c, h, w = x.size()
#         x = x.view(b, c, h * w)  # b * c * n
#         mu = self.mu.repeat(b, 1, 1)  # b * c * k
#         with torch.no_grad():
#             for i in range(self.stage_num):
#                 x_t = x.permute(0, 2, 1)  # b * n * c
#                 z = torch.bmm(x_t, mu)  # b * n * k
#                 z = F.softmax(z, dim=2)  # b * n * k
#                 z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
#                 mu = torch.bmm(x, z_)  # b * c * k
#                 mu = self._l2norm(mu, dim=1)
#
#         z_t = z.permute(0, 2, 1)  # b * k * n
#         x = mu.matmul(z_t)  # b * c * n
#         x = x.view(b, c, h, w)  # b * c * h * w
#         x = F.relu(x, inplace=True)
#
#         # The second 1x1 conv
#         x = self.conv2(x)
#         x = x + idn
#         x = F.relu(x, inplace=True)
#
#         return x
#
#     def _l2norm(self, inp, dim):
#         '''Normlize the inp tensor with l2-norm.
#         Returns a tensor where each sub-tensor of input along the given dim is
#         normalized such that the 2-norm of the sub-tensor is equal to 1.
#         Arguments:
#             inp (tensor): The input tensor.
#             dim (int): The dimension to slice over to get the ssub-tensors.
#         Returns:
#             (tensor) The normalized tensor.
#         '''
#         return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))
