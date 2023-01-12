import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# ======================================================
# 3d UNet
# ======================================================
class UNet3d(nn.Module):
    def __init__(self, in_channels=1, squeeze=False):
        super(UNet3d, self).__init__()
        n0 = 64 # was 64 initially
        self.conv1 = ConvBlock3d(in_channels, n0)
        self.down1 = DownConvBlock3d(n0, 2*n0)
        self.down2 = DownConvBlock3d(2*n0, 4*n0)
        self.down3 = DownConvBlock3d(4*n0, 8*n0)
        self.down4 = DownConvBlock3d(8*n0, 8*n0)
        self.up1 = UpConvBlock3d(8*n0, 4*n0)
        self.up2 = UpConvBlock3d(4*n0, 2*n0)
        self.up3 = UpConvBlock3d(2*n0, n0)
        self.up4 = UpConvBlock3d(n0, n0)
        self.out = OutConvBlock3d(n0, 1)
        self.squeeze = squeeze

    def forward(self, x):
        if self.squeeze:
            x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        if self.squeeze:
            x = x.squeeze(1)
        return x

# ======================================================
# A convolutional block consisting of two 3d conv layers, each followed by a 3d batch norm and a relu
# ======================================================
class ConvBlock3d(nn.Module):
    def __init__(self, in_size, out_size):
        super(ConvBlock3d, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=3, padding=1),
                                  nn.BatchNorm3d(out_size),
                                  nn.ReLU(inplace=True),
                                  nn.Conv3d(out_size, out_size, kernel_size=3, padding=1),
                                  nn.BatchNorm3d(out_size),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

# ======================================================
# A max-pool3d by a factor of 2, followed by the conv block defined above.
# ======================================================
class DownConvBlock3d(nn.Module):
    def __init__(self, in_size, out_size):
        super(DownConvBlock3d, self).__init__()
        self.down = nn.Sequential(nn.MaxPool3d(2),
                                  ConvBlock3d(in_size, out_size))

    def forward(self, x):
        x = self.down(x)
        return x

# ======================================================
# Takes two inputs.
# The first input is passed through a 3d transpose conv, 
# the output of this transpose conv is concatenated with the second input along the channel dimension
# this is now passed through the conv block defined above
# ======================================================
class UpConvBlock3d(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpConvBlock3d, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, in_size, kernel_size=2, stride=2)
        self.conv = ConvBlock3d(in_size * 2, out_size)

    def forward(self, x1, x2):
        up = self.up(x1)
        out = torch.cat([up, x2], dim=1)
        out = self.conv(out)
        return out

# ======================================================
# A 3d conv layer, without batch norm or activation function
# ======================================================
class OutConvBlock3d(nn.Module):
    def __init__(self, in_size, out_size):
        super(OutConvBlock3d, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.conv(x)
        return x

# ======================================================
# 2d UNet
# ======================================================
class UNet2d(nn.Module):

    def __init__(self,
                 in_channels = 1,
                 num_labels = 1,
                 squeeze = False,
                 returnlist = 2): # 1 logits, 2 features_and_logits

        super(UNet2d, self).__init__()
        n0 = 16        
        # downsampling and upsampling blocks / layers
        self.down = DownBlock2d()
        self.up = UpBlock2d()
        # conv blocks (each has two conv layers, followed by relu and batch norm)        
        self.enc_conv1 = ConvBlock2d(in_channels, n0)
        self.enc_conv2 = ConvBlock2d(n0, 2*n0)
        self.enc_conv3 = ConvBlock2d(2*n0, 4*n0)
        self.enc_conv4 = ConvBlock2d(4*n0, 8*n0)
        self.enc_conv5 = ConvBlock2d(8*n0, 8*n0)
        self.dec_conv1 = ConvBlock2d(16*n0, 4*n0)
        self.dec_conv2 = ConvBlock2d(8*n0, 2*n0)
        self.dec_conv3 = ConvBlock2d(4*n0, n0)
        self.dec_conv4 = ConvBlock2d(2*n0, n0)
        # final conv layer for segmentation (has one conv layer that outputs "num_labels" channels, no non-linearity)
        self.out_conv = OutConvBlock2d(n0, num_labels)
        # squeeze
        self.squeeze = squeeze
        # what to return
        self.returnlist = returnlist

    def forward(self, x):
        debugging = False
        if self.squeeze:
            x = x.unsqueeze(1)
        # encoder
        x1 = self.enc_conv1(x)
        if debugging: logging.info('x1' + str(x1.shape))
        x2 = self.down(x1)
        if debugging: logging.info('x2' + str(x2.shape))
        x3 = self.enc_conv2(x2)
        if debugging: logging.info('x3' + str(x3.shape))
        x4 = self.down(x3)
        if debugging: logging.info('x4' + str(x4.shape))
        x5 = self.enc_conv3(x4)
        if debugging: logging.info('x5' + str(x5.shape))
        x6 = self.down(x5)
        if debugging: logging.info('x6' + str(x6.shape))
        x7 = self.enc_conv4(x6)
        if debugging: logging.info('x7' + str(x7.shape))
        x8 = self.down(x7)
        if debugging: logging.info('x8' + str(x8.shape))
        x9 = self.enc_conv5(x8)
        if debugging: logging.info('x9' + str(x9.shape))
        # decoder
        x10 = self.up(x9)
        if debugging: logging.info('x10' + str(x10.shape))
        x11 = self.dec_conv1(torch.cat([x10, x7], dim=1))
        if debugging: logging.info('x11' + str(x11.shape))
        x12 = self.up(x11)
        if debugging: logging.info('x12' + str(x12.shape))
        x13 = self.dec_conv2(torch.cat([x12, x5], dim=1))
        if debugging: logging.info('x13' + str(x13.shape))
        x14 = self.up(x13)
        if debugging: logging.info('x14' + str(x14.shape))
        x15 = self.dec_conv3(torch.cat([x14, x3], dim=1))
        if debugging: logging.info('x15' + str(x15.shape))
        x16 = self.up(x15)
        if debugging: logging.info('x16' + str(x16.shape))
        x17 = self.dec_conv4(torch.cat([x16, x1], dim=1))
        if debugging: logging.info('x17' + str(x17.shape))
        # final layer
        x18 = self.out_conv(x17) # the outputs of this layer will be passed through a sigmoid function to get segmentation probabilities
        if debugging: logging.info('x18' + str(x18.shape))
        # squeeze
        if self.squeeze:
            x18 = x18.squeeze(1)

        if self.returnlist == 1:
            return [x18]
        else:
            return [x1, x3, x5, x7, x9, x11, x13, x15, x17, x18]

# ======================================================
# 2d UNet with heads for consistency reg at each layer
# ======================================================
class UNet2d_with_heads(nn.Module):
    def __init__(self,
                 in_channels = 1,
                 num_labels = 1,
                 squeeze = False,
                 returnlist = 1): # 1 heads_and_logits, 2 features_and_logits

        super(UNet2d_with_heads, self).__init__()        
        n0 = 16        
        # downsampling and upsampling blocks / layers
        self.down = DownBlock2d()
        self.up = UpBlock2d()
        # conv blocks (each has two conv layers, followed by relu and batch norm)        
        self.enc_conv1 = ConvBlock2d(in_channels, n0)
        self.enc_conv2 = ConvBlock2d(n0, 2*n0)
        self.enc_conv3 = ConvBlock2d(2*n0, 4*n0)
        self.enc_conv4 = ConvBlock2d(4*n0, 8*n0)
        self.enc_conv5 = ConvBlock2d(8*n0, 8*n0)
        self.dec_conv1 = ConvBlock2d(16*n0, 4*n0)
        self.dec_conv2 = ConvBlock2d(8*n0, 2*n0)
        self.dec_conv3 = ConvBlock2d(4*n0, n0)
        self.dec_conv4 = ConvBlock2d(2*n0, n0)
        # final conv layer for segmentation (has one conv layer that outputs "num_labels" channels, no non-linearity)
        self.out_conv = OutConvBlock2d(n0, num_labels)
        # helper blocks to define consistency at each layer
        self.head1 = HeadBlock2d(n0, n0)
        self.head2 = HeadBlock2d(2*n0, n0)
        self.head3 = HeadBlock2d(4*n0, n0)
        self.head4 = HeadBlock2d(8*n0, n0)
        self.head5 = HeadBlock2d(8*n0, n0)
        self.head6 = HeadBlock2d(4*n0, n0)
        self.head7 = HeadBlock2d(2*n0, n0)
        self.head8 = HeadBlock2d(n0, n0)
        self.head9 = HeadBlock2d(n0, n0)
        # squeeze
        self.squeeze = squeeze
        # what to return
        self.returnlist = returnlist

    def forward(self, x):
        debugging = False
        if self.squeeze:
            x = x.unsqueeze(1)
        # encoder
        x1 = self.enc_conv1(x)
        if debugging: logging.info('x1' + str(x1.shape))
        x2 = self.down(x1)
        if debugging: logging.info('x2' + str(x2.shape))
        x3 = self.enc_conv2(x2)
        if debugging: logging.info('x3' + str(x3.shape))
        x4 = self.down(x3)
        if debugging: logging.info('x4' + str(x4.shape))
        x5 = self.enc_conv3(x4)
        if debugging: logging.info('x5' + str(x5.shape))
        x6 = self.down(x5)
        if debugging: logging.info('x6' + str(x6.shape))
        x7 = self.enc_conv4(x6)
        if debugging: logging.info('x7' + str(x7.shape))
        x8 = self.down(x7)
        if debugging: logging.info('x8' + str(x8.shape))
        x9 = self.enc_conv5(x8)
        if debugging: logging.info('x9' + str(x9.shape))
        # decoder
        x10 = self.up(x9)
        if debugging: logging.info('x10' + str(x10.shape))
        x11 = self.dec_conv1(torch.cat([x10, x7], dim=1))
        if debugging: logging.info('x11' + str(x11.shape))
        x12 = self.up(x11)
        if debugging: logging.info('x12' + str(x12.shape))
        x13 = self.dec_conv2(torch.cat([x12, x5], dim=1))
        if debugging: logging.info('x13' + str(x13.shape))
        x14 = self.up(x13)
        if debugging: logging.info('x14' + str(x14.shape))
        x15 = self.dec_conv3(torch.cat([x14, x3], dim=1))
        if debugging: logging.info('x15' + str(x15.shape))
        x16 = self.up(x15)
        if debugging: logging.info('x16' + str(x16.shape))
        x17 = self.dec_conv4(torch.cat([x16, x1], dim=1))
        if debugging: logging.info('x17' + str(x17.shape))
        # final layer
        x18 = self.out_conv(x17) # the outputs of this layer will be passed through a sigmoid function to get segmentation probabilities
        if debugging: logging.info('x18' + str(x18.shape))
        # squeeze
        if self.squeeze:
            x18 = x18.squeeze(1)
        # heads
        h1 = self.head1(x1)
        if debugging: logging.info('h1' + str(h1.shape))
        h2 = self.head2(x3)
        if debugging: logging.info('h2' + str(h2.shape))
        h3 = self.head3(x5)
        if debugging: logging.info('h3' + str(h3.shape))
        h4 = self.head4(x7)
        if debugging: logging.info('h4' + str(h4.shape))
        h5 = self.head5(x9)
        if debugging: logging.info('h5' + str(h5.shape))
        h6 = self.head6(x11)
        if debugging: logging.info('h6' + str(h6.shape))
        h7 = self.head7(x13)
        if debugging: logging.info('h7' + str(h7.shape))
        h8 = self.head8(x15)
        if debugging: logging.info('h8' + str(h8.shape))
        h9 = self.head9(x17)
        if debugging: logging.info('h9' + str(h9.shape))

        if self.returnlist == 1:
            return [h1, h2, h3, h4, h5, h6, h7, h8, h9, x18]
        else:
            return [x1, x3, x5, x7, x9, x11, x13, x15, x17, x18]

# ======================================================
# A convolutional block of the head after which the consistency will be computed
# ======================================================
class HeadBlock2d(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(HeadBlock2d, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(in_size, hidden_size, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(hidden_size),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.head(x)
        return x

# ======================================================
# A convolutional block consisting of two 2d conv layers, each followed by a 2d batch norm and a relu
# ======================================================
class ConvBlock2d(nn.Module):
    def __init__(self, in_size, out_size):
        super(ConvBlock2d, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_size),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_size),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

# ======================================================
# A max-pool2d by a factor of 2
# ======================================================
class DownBlock2d(nn.Module):
    def __init__(self):
        super(DownBlock2d, self).__init__()
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.down(x)
        return x

# ======================================================
# upsampling layer
# ======================================================
class UpBlock2d(nn.Module):
    def __init__(self):
        super(UpBlock2d, self).__init__()
        # self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        up = self.up(x)
        return up

# ======================================================
# A 2d conv layer, without batch norm or activation function
# ======================================================
class OutConvBlock2d(nn.Module):
    def __init__(self, in_size, out_size):
        super(OutConvBlock2d, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.conv(x)
        return x