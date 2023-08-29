import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_parts import *

# Define the neural networks
class my_UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(my_UNet, self).__init__()

        # Encoder
        self.enc_conv1 = self.conv_block(n_channels, 64)
        self.enc_conv2 = self.conv_block(64, 128)
        self.enc_conv3 = self.conv_block(128, 256)
        self.enc_conv4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = self.conv_block(128, 64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            #nn.Sigmoid()
        )

    def conv_block(self, in_channels, out_channels, mid_channels=None):
        if not mid_channels:
            mid_channels = out_channels
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
       
        # Encoding
        enc1 = self.enc_conv1(x)
        x = self.pool(enc1)
        enc2 = self.enc_conv2(x)
        x = self.pool(enc2)
        enc3 = self.enc_conv3(x)
        x = self.pool(enc3)
        enc4 = self.enc_conv4(x)

        # Decoding
        x = self.upconv3(enc4)
        x = torch.cat((x, enc3), dim=1)
        x = self.dec_conv3(x)
        x = self.upconv2(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.dec_conv2(x)
        x = self.upconv1(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.dec_conv1(x)
        x = self.final_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)