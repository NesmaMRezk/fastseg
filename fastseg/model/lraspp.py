"""Lite Reduced Atrous Spatial Pyramid Pooling

Architecture introduced in the MobileNetV3 (2019) paper, as an
efficient semantic segmentation head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_trunk, ConvBnRelu
from .base import BaseSegmentation

class LRASPP(BaseSegmentation):
    """Lite R-ASPP style segmentation network."""
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

        # Reduced atrous spatial pyramid pooling
        if self.use_aspp:
            self.aspp_conv1 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv2 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Conv2d(num_filters, num_filters, 3, dilation=12, padding=12),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv3 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Conv2d(num_filters, num_filters, 3, dilation=36, padding=36),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
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

        self.convs2 = nn.Conv2d(s2_ch, 32, kernel_size=1, bias=False)
        self.convs4 = nn.Conv2d(s4_ch, 64, kernel_size=1, bias=False)
        self.conv_up1 = nn.Conv2d(aspp_out_ch, num_filters, kernel_size=1)
        self.conv_up2 = ConvBnRelu(num_filters + 64, num_filters, kernel_size=1)
        self.conv_up3 = ConvBnRelu(num_filters + 32, num_filters, kernel_size=1)
        self.last = nn.Conv2d(num_filters, num_classes, kernel_size=1)

        # ConvTranspose2d layers to replace F.interpolate
        self.upsample1 = nn.ConvTranspose2d(num_filters, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(num_filters, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(num_filters, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        s2, s4, final = self.trunk(x), self.trunk(x), self.trunk(x)  # Simulating trunk outputs
        if self.use_aspp:
            aspp_features = [
                self.aspp_conv1(final),
                self.aspp_conv2(final),
                self.aspp_conv3(final),
                self.upsample1(self.aspp_pool(final))  # Upsample with ConvTranspose2d
            ]
            aspp = torch.cat(aspp_features, 1)
        else:
            aspp = self.aspp_conv1(final) * self.upsample1(self.aspp_conv2(final))

        y = self.conv_up1(aspp)

        # Adjust the spatial dimensions to match `s4`
        y = self.upsample2(y)
        y = torch.cat([y, self.convs4(s4)], 1)
        y = self.conv_up2(y)
        
        # Adjust the spatial dimensions to match `s2`
        y = self.upsample3(y)
        y = torch.cat([y, self.convs2(s2)], 1)
        y = self.conv_up3(y)
        
        # Perform the final convolution without resizing
        y = self.last(y)

        return y

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
