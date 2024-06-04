"""Lite Reduced Atrous Spatial Pyramid Pooling

Architecture introduced in the MobileNetV3 (2019) paper, as an
efficient semantic segmentation head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_trunk, ConvBnRelu
from .base import BaseSegmentation

import torch.nn as nn

class LRASPP(nn.Module):
    def __init__(self, in_channels, out_channels, aspp_channels=64):
        super(LRASPP, self).__init__()

        # ASPP modules
        self.aspp_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, aspp_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, aspp_channels, kernel_size=1, bias=False),
            nn.Conv2d(aspp_channels, aspp_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, aspp_channels, kernel_size=1, bias=False),
            nn.Conv2d(aspp_channels, aspp_channels, kernel_size=3, stride=1, padding=24, dilation=24, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, aspp_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True)
        )

        # Convolution to reduce the number of channels
        self.convs2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.convs4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # Upsampling layers using strided convolutions
        self.upsample1 = nn.ConvTranspose2d(aspp_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(aspp_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(aspp_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # ASPP module
        aspp_features = [
            self.aspp_conv1(x),
            self.aspp_conv2(x),
            self.aspp_conv3(x),
            self.aspp_pool(x)
        ]
        aspp = torch.cat(aspp_features, 1)

        # Reduce the number of channels
        s2 = self.convs2(x)
        s4 = self.convs4(x)

        # Upsampling
        up1 = self.upsample1(aspp)
        up2 = self.upsample2(aspp)
        up3 = self.upsample3(aspp)

        return s2, s4, up1, up2, up3

class MobileV3Large(LRASPP):
    """MobileNetV3-Large segmentation network."""
    model_name = 'mobilev3large-lraspp'

    def __init__(self, num_classes, **kwargs):
        super(MobileV3Large, self).__init__(
            num_classes,
            trunk='mobilenetv3_large',
            **kwargs
        )


class MobileV3Small(LRASPP):
    """MobileNetV3-Small segmentation network."""
    model_name = 'mobilev3small-lraspp'

    def __init__(self, num_classes, **kwargs):
        super(MobileV3Small, self).__init__(
            num_classes,
            trunk='mobilenetv3_small',
            **kwargs
        )
