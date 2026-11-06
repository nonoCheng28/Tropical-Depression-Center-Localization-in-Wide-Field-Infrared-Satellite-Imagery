# Adapted from chao-tan/TCLNet, Copyright (c) 2020 chao-tan.
# MIT notice: third_party/TCLNet-LICENSE.
"""TCLNet (Tan, IGARSS 2021) adapted to this project's comparison protocol.

The architecture below is the published one, copied operation for operation from
`github.com/chao-tan/TCLNet` (`modules/components/TCLNET.py`, MIT licence). Two
adaptations are made and both are disclosed:

1. **Input.** The published net starts with `ConvBlock(3, 16, 7, 2)`. Our IR is a
   single channel, so the frame is replicated three times inside `forward()`
   instead of editing the first convolution. No other layer changes.
2. **Supervision.** The published net emits a same-size map. We train it against
   our 80x80, sigma = 2 gaussian target with the project's weighted MSE
   (alpha = 100), so the loss target is ours and the network is theirs.

A 320x320 input produces exactly an 80x80 map, because the stem downsamples by 2 and
the three pooled stages plus three upsamplings bring it back to input/4. That is why
this model drops into the existing decoder unchanged.

This is *not* a reproduction of the paper's number: TCLD is 666x666 centred patches
scored in km, ours is a 70x70 degree full field scored in degrees, and one heatmap
cell (0.875 deg) is coarser than the paper's whole reported error (20.1 km = 0.18 deg).
Only the internal comparison is meaningful.
"""
from torch import nn


class ConvBlock(nn.Module):
    """conv-bn-relu, the published helper."""

    def __init__(self, inp_dim, out_dim, kernel_size, stride, use_bn, use_relu):
        super().__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
                              padding=(kernel_size - 1) // 2, bias=True)
        self.relu = nn.ReLU(inplace=True) if use_relu else None
        self.bn = nn.BatchNorm2d(out_dim) if use_bn else None

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, f"{x.size()[1]} {self.inp_dim}"
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ResBlock(nn.Module):
    """Pre-activation bottleneck residual block, the published helper."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = ConvBlock(in_dim, int(out_dim / 2), 1, stride=1, use_bn=False, use_relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = ConvBlock(int(out_dim / 2), int(out_dim / 2), 3, stride=1, use_bn=False,
                               use_relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = ConvBlock(int(out_dim / 2), out_dim, 1, stride=1, use_bn=False, use_relu=False)
        self.expand_conv = ConvBlock(in_dim, out_dim, 1, 1, False, False) if in_dim != out_dim else None

    def forward(self, x):
        residual = self.expand_conv(x) if self.expand_conv is not None else x
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        out = self.relu(self.bn3(out))
        out = self.conv3(out)
        return out + residual


class TCLNetModel(nn.Module):
    """[B, 1, 320, 320] -> [B, 1, 80, 80]; the published topology, single-channel input."""

    def __init__(self):
        super().__init__()
        self.maxpooling = nn.MaxPool2d(2, 2)
        self.upsampling = nn.Upsample(scale_factor=2)
        self.pre = nn.Sequential(
            ConvBlock(3, 16, 7, 2, use_bn=True, use_relu=True),
            ConvBlock(16, 32, 1, 1, use_bn=False, use_relu=False),
            ResBlock(32, 32),
            nn.MaxPool2d(2, 2),
            ResBlock(32, 32),
            ConvBlock(32, 64, 1, 1, use_bn=False, use_relu=False),
            ResBlock(64, 64),
        )
        self.down1 = ResBlock(64, 128)
        self.down2 = ResBlock(128, 256)
        self.down3 = ResBlock(256, 256)
        self.inner = ResBlock(256, 256)
        self.up3 = ResBlock(256, 256)
        self.up2 = ResBlock(256, 128)
        self.up1 = ResBlock(128, 64)
        self.outter = nn.Sequential(
            ResBlock(64, 64),
            ConvBlock(64, 64, 1, 1, use_bn=True, use_relu=True),
            ConvBlock(64, 1, 1, 1, use_bn=False, use_relu=False),
        )

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 320, 320):
            raise ValueError("TCLNet 输入必须为 [B, 1, 320, 320] 的单帧 IR")
        x = images.repeat(1, 3, 1, 1)          # grayscale -> 3 channels, published stem expects 3
        x = self.pre(x)
        x = self.maxpooling(self.down1(x))
        x = self.maxpooling(self.down2(x))
        x = self.maxpooling(self.down3(x))
        x = self.inner(x)
        x = self.upsampling(self.up3(x))
        x = self.upsampling(self.up2(x))
        x = self.upsampling(self.up1(x))
        return self.outter(x)
