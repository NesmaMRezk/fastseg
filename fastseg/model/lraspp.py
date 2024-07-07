import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorly.decomposition import partial_tucker
from .utils import get_trunk, ConvBnRelu
from .base import BaseSegmentation
#tucker applied to bottlneck conv layers
import tensorly as tl


# Function to apply Tucker decomposition to a convolutional layer
def tucker_decompose_conv_layer(layer, rank):
    weight = layer.weight.data
    in_channels, out_channels, k_h, k_w = weight.shape
    rank = (rank, rank, k_h, k_w)  # Adjust rank to match tensor dimensions
    core_all, factors = partial_tucker(weight, rank=rank, modes=[0, 1])
    #core=core_all[0]
    core, [*factors] = core_all
    print(factors)
    pointwise_s_to_r = nn.Conv2d(in_channels=in_channels, out_channels=core.shape[0],
                                 kernel_size=1, stride=1, padding=0, bias=False)
    pointwise_s_to_r.weight.data = factors[0].unsqueeze(2).unsqueeze(3)

    depthwise_r_to_r = nn.Conv2d(in_channels=core.shape[0], out_channels=core.shape[0],
                                 kernel_size=(k_h, k_w), stride=layer.stride,
                                 padding=layer.padding, dilation=layer.dilation,
                                 groups=core.shape[0], bias=False)
    depthwise_r_to_r.weight.data = core

    pointwise_r_to_t = nn.Conv2d(in_channels=core.shape[0], out_channels=out_channels,
                                 kernel_size=1, stride=1, padding=0, bias=False)
    pointwise_r_to_t.weight.data = factors[1].unsqueeze(2).unsqueeze(3)

    decomposed_layer = nn.Sequential(
        pointwise_s_to_r,
        depthwise_r_to_r,
        pointwise_r_to_t
    )

    return decomposed_layer


class LRASPP_tenosr(BaseSegmentation):
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

        print("helllo  2")
        # Apply Tucker decomposition to the segmentation head
        self.conv_up1 = tucker_decompose_conv_layer(nn.Conv2d(aspp_out_ch, num_filters, kernel_size=1), rank=64)
        self.conv_up2 = tucker_decompose_conv_layer(ConvBnRelu(num_filters + 64, num_filters, kernel_size=1).conv, rank=64)
        self.conv_up3 = tucker_decompose_conv_layer(ConvBnRelu(num_filters + 32, num_filters, kernel_size=1).conv, rank=64)
        self.last = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        
    def forward(self, x):
        s2, s4, final = self.trunk(x)
        if self.use_aspp:
            aspp = torch.cat([
                self.aspp_conv1(final),
                self.aspp_conv2(final),
                self.aspp_conv3(final),
                F.interpolate(self.aspp_pool(final), size=final.shape[2:], mode='bilinear', align_corners=True),
            ], 1)
        else:
            aspp = self.aspp_conv1(final) * F.interpolate(
                self.aspp_conv2(final),
                final.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        y = self.conv_up1(aspp)
        y = F.interpolate(y, size=s4.shape[2:], mode='bilinear', align_corners=False)

        y = torch.cat([y, self.convs4(s4)], 1)
        y = self.conv_up2(y)
        y = F.interpolate(y, size=s2.shape[2:], mode='bilinear', align_corners=False)

        y = torch.cat([y, self.convs2(s2)], 1)
        y = self.conv_up3(y)
        y = self.last(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y
        
def adjust_size(y, target_size):
    _, _, h, w = y.size()
    target_h, target_w = target_size

    # Calculate the padding needed
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)

    # Apply padding if needed
    y = F.pad(y, (0, pad_w, 0, pad_h))

    # Crop to the target size
    y = y[:, :, :target_h, :target_w]

    return y
    
# CustomUpsampleLayer definition
class CustomUpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomUpsampleLayer, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
    
    def forward(self, x, output_size=None):
        return self.upconv(x, output_size=output_size)
import torch.nn.functional as F

class LRASPP_repeat_notworking(BaseSegmentation):
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
        self.conv_after_repeat = nn.Conv2d(num_filters, num_filters, kernel_size=1)
        self.last = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        s2, s4, final = self.trunk(x)
        if self.use_aspp:
            aspp = torch.cat([
                self.aspp_conv1(final),
                self.aspp_conv2(final),
                self.aspp_conv3(final),
                self.conv_after_repeat(self.aspp_pool(final).repeat(1, 1,final.shape[2], final.shape[3])),], 1)
        else:
            aspp = self.aspp_conv1(final) * F.interpolate(
                self.aspp_conv2(final),
                final.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        y = self.conv_up1(aspp)
       # y = F.interpolate(y, size=s4.shape[2:], mode='bilinear', align_corners=False)
       # Calculate the repeating factors
        repeat_factor_h = s4.size(2) // y.size(2) + 1  # How many times to repeat in height
        repeat_factor_w = s4.size(3) // y.size(3) + 1  # How many times to repeat in width
        
        # Repeat the tensor
        y_repeated = y.repeat(1, 1, repeat_factor_h, repeat_factor_w)
        
        # Now y_repeated might be larger than needed, so we slice it to match the desired shape
        y= y_repeated[:, :, :s4.size(2), :s4.size(3)]
        y = torch.cat([self.conv_after_repeat(y), self.convs4(s4)], 1)
        y = self.conv_up2(y)
        #y = F.interpolate(y, size=s2.shape[2:], mode='bilinear', align_corners=False)
       # y=y.repeat(1,1,s2.size(2),s2.size(3))
        repeat_factor_h = s2.size(2) // y.size(2) + 1  # How many times to repeat in height
        repeat_factor_w = s2.size(3) // y.size(3) + 1  # How many times to repeat in width
        y_repeated = y.repeat(1, 1, repeat_factor_h, repeat_factor_w)
        y= y_repeated[:, :, :s2.size(2), :s2.size(3)]
        
        y = torch.cat([self.conv_after_repeat(y), self.convs2(s2)], 1)
        y = self.conv_up3(y)
        y = self.last(y)
        #y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        #y_r=y.repeat(1,1,x.size(2),x.size(3))
        #y = y_r[:, :, :desired_shape[2], :desired_shape[3]]
        repeat_factor_h = x.size(2) // x.size(2) + 1  # How many times to repeat in height
        repeat_factor_w = x.size(3) // x.size(3) + 1  # How many times to repeat in width
        y_repeated = y.repeat(1, 1, repeat_factor_h, repeat_factor_w)
        y= y_repeated[:, :, :x.size(2), :x.size(3)]
        y=self.conv_after_repeat(y)
        return y


tl.set_backend('pytorch')

# Define a helper class for ConvBnRelu
class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


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

    def forward(self, x):
        s2, s4, final = self.trunk(x)
        print(final.dim())
        if self.use_aspp:
            aspp = torch.cat([
                self.aspp_conv1(final),
                self.aspp_conv2(final),
                self.aspp_conv3(final),
                F.interpolate(self.aspp_pool(final), size=final.shape[2:],mode='bilinear', align_corners=True),
            ], 1)
        else:
            aspp = self.aspp_conv1(final) * F.interpolate(
                self.aspp_conv2(final),
                final.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        y = self.conv_up1(aspp)
        y = F.interpolate(y, size=s4.shape[2:], mode='bilinear', align_corners=False)

        y = torch.cat([y, self.convs4(s4)], 1)
        y = self.conv_up2(y)
        y = F.interpolate(y, size=s2.shape[2:], mode='bilinear', align_corners=False)

        y = torch.cat([y, self.convs2(s2)], 1)
        y = self.conv_up3(y)
        y = self.last(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y
class LRASPP_no_interpolate2(BaseSegmentation):
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

        self.trunk, _, _, high_level_ch = get_trunk(trunk_name=trunk)
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

        self.conv_up1 = nn.Conv2d(aspp_out_ch, num_filters, kernel_size=1)
        self.conv_up2 = ConvBnRelu(num_filters, num_filters, kernel_size=1)
        self.conv_up3 = ConvBnRelu(num_filters, num_filters, kernel_size=1)
        self.last = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        _, _, final = self.trunk(x)
        if self.use_aspp:
            aspp = torch.cat([
                self.aspp_conv1(final),
                self.aspp_conv2(final),
                self.aspp_conv3(final),
                self.aspp_pool(final).expand(-1, -1, final.size(2), final.size(3)),
            ], 1)
        else:
            aspp = self.aspp_conv1(final) * self.aspp_conv2(final)

        # Perform operations to ensure the output size matches the input size
        # (No interpolation needed)

        # Apply the segmentation head layers
        y = self.conv_up1(aspp)
        y = self.conv_up2(y)
        y = self.conv_up3(y)
        y = self.last(y)
        
        return y


class LRASPP_no_interpolate(BaseSegmentation):
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
        self.conv_up1 = CustomUpsampleLayer(aspp_out_ch, num_filters)
        self.conv_up2 = CustomUpsampleLayer(num_filters + 64, num_filters)
        self.conv_up3 = CustomUpsampleLayer(num_filters + 32, num_filters)
        self.last = nn.Conv2d(num_filters, num_classes, kernel_size=1)
   
    def forward(self, x):
        s2, s4, final = self.trunk(x)
        if self.use_aspp:
            aspp = torch.cat([
                self.aspp_conv1(final),
                self.aspp_conv2(final),
                self.aspp_conv3(final),
                self.aspp_pool(final).expand(-1, -1, final.size(2), final.size(3)),
            ], 1)
        else:
            aspp = self.aspp_conv1(final) * F.interpolate(
                self.aspp_conv2(final),
                final.shape[2:],
                mode='bilinear',
                align_corners=True
            )
    
        # Pad the tensors to adjust sizes
        y = self.conv_up1(aspp, output_size=s4.shape[2:])
        y = torch.cat([y, self.convs4(s4)], 1)
        y = self.conv_up2(y, output_size=s2.shape[2:])
        y = torch.cat([y, self.convs2(s2)], 1)
        y = self.conv_up3(y, output_size=x.shape[2:])
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
