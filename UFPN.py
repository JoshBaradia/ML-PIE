import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models import vgg19

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.enc_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(64)
        self.enc_gelu1 = nn.GELU()

        self.enc_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_gelu2 = nn.GELU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(64)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 64,64],
            out_channels=256,
        )

        self.middle_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.middle_bn1 = nn.BatchNorm2d(128)
        self.middle_gelu1 = nn.GELU()

        self.middle_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.middle_bn2 = nn.BatchNorm2d(128)
        self.middle_gelu2 = nn.GELU()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec_conv1 = nn.Conv2d(192, 64, kernel_size=3, padding=1)  # Concatenation increases channels to 192
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.dec_gelu1 = nn.GELU()

        self.dec_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.dec_bn2 = nn.BatchNorm2d(64)
        self.dec_gelu2 = nn.GELU()

        self.dec_conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        # self.dec_bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        enc1 = self.enc_gelu1(self.enc_bn1(self.enc_conv1(x)))
        # print("enc1: ", torch.isnan(enc1).any())
        assert not torch.isnan(enc1).any(), "NaN value found in enc1"

        enc2 = self.enc_gelu2(self.enc_bn2(self.enc_conv2(enc1)))
        # print("enc2: ", torch.isnan(enc2).any())
        assert not torch.isnan(enc2).any(), "NaN value found in enc2"

        pooled = self.pool(enc2)

        spatial_attention = self.spatial_attention(pooled)
        channel_attention = self.channel_attention(pooled)
        attention = spatial_attention * channel_attention
        fpn_out = self.fpn({'enc1': enc1,'enc2': enc2, 'attention': attention})

        mid1 = self.middle_gelu1(((self.middle_bn1(self.middle_conv1(fpn_out['enc1'])))))
        # print("mid1: ", torch.isnan(mid1).any())
        assert not torch.isnan(mid1).any(), "NaN value found in mid1"

        mid2 = self.middle_gelu2((self.middle_conv2(mid1)))
        # print("mid2: ", torch.isnan(mid2).any())
        assert not torch.isnan(mid2).any(), "NaN value found in mid2"

        upsampled = self.upsample(mid2)
        # print("upsampled: ", torch.isnan(upsampled).any())
        assert not torch.isnan(upsampled).any(), "NaN value found in upsampled"

        # Adjust the size of 'upsampled' to match 'enc2'
        if upsampled.shape != enc2.shape:
            upsampled = F.interpolate(upsampled, size=enc2.shape[2:], mode='bilinear', align_corners=False)

        concatenated = torch.cat((upsampled, enc2), dim=1)  # Skip connection
        # print("concatenated: ", torch.isnan(concatenated).any())
        assert not torch.isnan(concatenated).any(), "NaN value found in concatenated"

        dec1 = self.dec_gelu1((self.dec_bn1(self.dec_conv1(concatenated))))
        # print("dec1: ", torch.isnan(dec1).any())
        assert not torch.isnan(dec1).any(), "NaN value found in dec1"

        dec2 = self.dec_gelu2((self.dec_conv2(dec1)))
        # print("dec2: ", torch.isnan(dec2).any())
        assert not torch.isnan(dec2).any(), "NaN value found in dec2"

        dec3 = self.dec_conv3(dec2)
        # print("dec3: ", torch.isnan(dec3).any())
        assert not torch.isnan(dec3).any(), "NaN value found in dec3"

        return dec3

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval()  # Use the features up to the 36th layer of VGG19
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze the VGG model
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        perceptual_loss = F.l1_loss(self.vgg(x), self.vgg(y))
        l1_loss = self.l1_loss(x, y)
        return perceptual_loss + l1_loss

# class CombinedLoss(nn.Module):
#     def __init__(self):
#         super(CombinedLoss, self).__init__()
#         self.vgg = vgg19(pretrained=True).features[:36].eval()  # Use the features up to the 36th layer of VGG19
#         for param in self.vgg.parameters():
#             param.requires_grad = False  # Freeze the VGG model
#         self.huber_loss = nn.SmoothL1Loss()  # Use Huber loss (SmoothL1Loss in PyTorch)

#     def forward(self, x, y):
#         perceptual_loss = F.l1_loss(self.vgg(x), self.vgg(y))
#         huber_loss = self.huber_loss(x, y)
#         return perceptual_loss + huber_loss


class EnhancementNet(nn.Module):
    def __init__(self):
        super(EnhancementNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        # self.init_weights()
        self.unet = UNet(3, 3)
        # self.unet.apply(self.init_weights) 
        self.combined_loss = CombinedLoss()
    
    # def init_weights(self, m):
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight)

    def forward(self, x, target=None):
        x = self.unet(x)
        if target is not None:
            # Calculate the loss if a target is provided
            loss = self.combined_loss(x, target)
            return x, loss
        else:
            # If no target is provided, just return the output
            return x
        # loss = self.combined_loss(x, target)
        # return x, loss
