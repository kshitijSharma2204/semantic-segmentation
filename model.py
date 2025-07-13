import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.dconv_down1 = DoubleConv(3, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.ConvTranspose2d(512, 512, 2, stride=2)

        self.dconv_up3 = DoubleConv(512 + 256, 256)
        self.dconv_up2 = DoubleConv(256 + 128, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
      conv1 = self.dconv_down1(x)
      x = self.maxpool(conv1)
      conv2 = self.dconv_down2(x)
      x = self.maxpool(conv2)
      conv3 = self.dconv_down3(x)
      x = self.maxpool(conv3)
      x = self.dconv_down4(x)
      x = self.upsample(x)

      # CROP conv3 to match x
      if x.shape[2:] != conv3.shape[2:]:
          conv3 = self.center_crop(conv3, x.shape[2:])
      x = torch.cat([x, conv3], dim=1)
      x = self.dconv_up3(x)
      x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

      if x.shape[2:] != conv2.shape[2:]:
          conv2 = self.center_crop(conv2, x.shape[2:])
      x = torch.cat([x, conv2], dim=1)
      x = self.dconv_up2(x)
      x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

      if x.shape[2:] != conv1.shape[2:]:
          conv1 = self.center_crop(conv1, x.shape[2:])
      x = torch.cat([x, conv1], dim=1)
      x = self.dconv_up1(x)
      out = self.conv_last(x)
      return out
    
    def center_crop(self, enc_feat, target_shape):
      _, _, h, w = enc_feat.shape
      th, tw = target_shape
      dh = (h - th) // 2
      dw = (w - tw) // 2
      return enc_feat[:, :, dh:dh+th, dw:dw+tw]
