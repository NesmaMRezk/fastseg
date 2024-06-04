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
     def __init__(self, num_classes, trunk, use_aspp=False, num_filters=128):
        """Initialize a new segmentation model.

        Keyword arguments:
        num_classes -- number of output classes (e.g., 19 for Cityscapes)
        trunk -- the name of the trunk to use ('mobilenetv3_large', 'mobilenetv3_small')
        use_aspp -- whether to use DeepLabV3+ style ASPP (True) or Lite R-ASPP (False)
            (setting this to True may yield better results, at the cost of latency)
        num_filters -- the number of filters in the segmentation head
        """
        super(LRASPP, self).__init__()

        self.trunk, s2_ch, s4_ch, high_level_ch = get_trunk(trunk_name=trunk)
        self.use_aspp = use_aspp
         
        # ASPP modules
        if self.use_aspp:
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
            aspp_out_ch = num_filters * 4
        else:
            self.aspp_conv1 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv2 = nn.Sequential(
                nn.AvgPool2d(kernel_size=(49, 49), stride=(16, 20)),
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Sigmoid(),
            )
            aspp_out_ch = num_filters
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
