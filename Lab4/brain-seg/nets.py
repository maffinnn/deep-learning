from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


############################################################################################
############################################################################################

class UNetSkipConn(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNetSkipConn, self).__init__()

        features = init_features
        self.encoder1 = UNetSkipConn._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut1 = UNetSkipConn._conv1x1(in_channels, features)
        
        self.encoder2 = UNetSkipConn._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut2 = UNetSkipConn._conv1x1(features, features*2)
        
        self.encoder3 = UNetSkipConn._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut3 = UNetSkipConn._conv1x1(features*2, features*4)

        self.encoder4 = UNetSkipConn._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut4 = UNetSkipConn._conv1x1(features*4, features*8)
  
        self.bottleneck = UNetSkipConn._block(features * 8, features * 16, name="bottleneck")
        self.shortcut5 = UNetSkipConn._conv1x1(features*8, features*16)
   
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNetSkipConn._block((features * 8) * 2, features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNetSkipConn._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNetSkipConn._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNetSkipConn._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):

        enc1 = self.encoder1(x)
        enc1 = F.relu(enc1 + self.shortcut1(x))
        
        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = F.relu(enc2 + self.shortcut2(self.pool1(enc1)))

        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = F.relu(enc3 + self.shortcut3(self.pool2(enc2)))
        
        enc4 = self.encoder4(self.pool3(enc3))
        enc4 = F.relu(enc4 + self.shortcut4(self.pool3(enc3)))

        bottleneck = self.bottleneck(self.pool4(enc4))
        bottleneck = F.relu(bottleneck + self.shortcut5(self.pool4(enc4)))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = F.relu(dec4 + self.upconv4(bottleneck)) # shortcut6

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec3 = F.relu(dec3 + self.upconv3(dec4)) # shortcut7

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec2 = F.relu(dec2 + self.upconv2(dec3)) #shortcut8

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = F.relu(dec1 + self.upconv1(dec2)) #shortcut9
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )
            
    @staticmethod
    def _conv1x1(in_channels, out_channels): #projection shortcut
        return nn.Conv2d(in_channels, out_channels, 1) # kernel size = 1


###################################################################################
###################################################################################


class FPN(nn.Module):

    def __init__(self, n_classes=1, pyramid_channels=256, segmentation_channels=256):
        super().__init__()
         
        # Bottom-up layers
        self.encoder1 = UNet._block(3, 64, "enc1")
        self.encoder2 = UNet._block(64, 128, "enc2")
        self.encoder3 = UNet._block(128, 256, "enc3")
        self.encoder4 = UNet._block(256, 512, "enc4")        
        self.encoder5 = UNet._block(512, 1024, "enc5")   
        self.maxpool = nn.MaxPool2d(2)
        
        # Top layer
        self.top = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.lat1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.lat2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.lat3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        
        # Segmentation block layers
        self.seg_blocks = nn.ModuleList([
            FPN._SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0),
            FPN._SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1),
            FPN._SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2),
            FPN._SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3),
        ])
        # Last layer
        self.last_conv = nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0)
    
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        _upsample = nn.Upsample(size=(H,W), mode='bilinear', align_corners=True) 
        return _upsample(x) + y
        
    def _upsample(self, x, h, w):
        sample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
        return sample(x)
        
    def forward(self, x):
        # Bottom-up
        enc1 = self.maxpool(self.encoder1(x))
        enc2 = self.maxpool(self.encoder2(enc1))
        enc3 = self.maxpool(self.encoder3(enc2))
        enc4 = self.maxpool(self.encoder4(enc3))
        enc5 = self.maxpool(self.encoder5(enc4)) 
        
        # Top-down
        p5 = self.top(enc5) 
        p4 = self._upsample_add(p5, self.lat1(enc4)) 
        p3 = self._upsample_add(p4, self.lat2(enc3))
        p2 = self._upsample_add(p3, self.lat3(enc2)) 
        
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        
        # Segmentation
        _, _, h, w = p2.size()
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p2, p3, p4, p5])]
        out = self._upsample(self.last_conv(sum(feature_pyramid)), 4 * h, 4 * w)
        out = torch.sigmoid(out)
        return out

    @staticmethod
    def _ConvReluUpsample(in_channels, out_channels, upsample=False):
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.GroupNorm(32, out_channels),
                  nn.ReLU(inplace=True),]
        if upsample:
          modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        return nn.Sequential(*modules)

    @staticmethod
    def _SegmentationBlock(in_channels, out_channels, n_upsamples=0):
        blocks = [FPN._ConvReluUpsample(in_channels, out_channels, upsample=bool(n_upsamples))]
        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(FPN._ConvReluUpsample(out_channels, out_channels, upsample=True))
        return nn.Sequential(*blocks)


##########################################################################################
##########################################################################################



class ResNeXtUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        self.base_model = resnext50_32x4d(pretrained=True)
        self.base_layers = list(self.base_model.children())
        filters = [4*64, 4*128, 4*256, 4*512]
        
        # Down
        self.encoder0 = nn.Sequential(*self.base_layers[:3])
        self.encoder1 = nn.Sequential(*self.base_layers[4])
        self.encoder2 = nn.Sequential(*self.base_layers[5])
        self.encoder3 = nn.Sequential(*self.base_layers[6])
        self.encoder4 = nn.Sequential(*self.base_layers[7])

        # Up
        self.decoder4 = ResNeXtUNet._DecoderBlock(filters[3], filters[2])
        self.decoder3 = ResNeXtUNet._DecoderBlock(filters[2], filters[1])
        self.decoder2 = ResNeXtUNet._DecoderBlock(filters[1], filters[0])
        self.decoder1 = ResNeXtUNet._DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.last_conv0 = ResNeXtUNet._ConvRelu(256, 128, 3, 1)
        self.last_conv1 = nn.Conv2d(128, n_classes, 3, padding=1)
                       
        
    def forward(self, x):
        # Down
        x = self.encoder0(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Up + sc
        dec4 = self.decoder4(enc4) + enc3
        dec3 = self.decoder3(dec4) + enc2
        dec2 = self.decoder2(dec3) + enc1
        dec1 = self.decoder1(dec2)
        #print(d1.shape)

        # final classifier
        out = self.last_conv0(dec1)
        out = self.last_conv1(out)
        out = torch.sigmoid(out)
        return out

    @staticmethod
    def _ConvRelu(in_channels, out_channels, kernel, padding):
        return  nn.Sequential( nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
                      nn.ReLU(inplace=True))
    @staticmethod
    def _DecoderBlock(in_channels, out_channels):
        return nn.Sequential(ResNeXtUNet._ConvRelu(in_channels, in_channels // 4, 1, 0),
                              nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4, 
                                    stride=2, padding=1, output_padding=0),
                              ResNeXtUNet._ConvRelu(in_channels // 4, out_channels, 1, 0))

