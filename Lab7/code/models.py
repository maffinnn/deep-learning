import torch
from torch import nn
import torchvision
import math

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        n_channels = config.G.n_channels
        n_blocks = config.G.n_blocks
        large_kernel_size = config.G.large_kernel_size
        small_kernel_size = config.G.small_kernel_size

        upsample_block_num = int(math.log(config.scaling_factor, 2))
        self.conv1 = nn.Conv2d(3, out_channels=n_channels, kernel_size=large_kernel_size, padding=4)
        self.prelu = nn.PReLU()

        block1 = [_residualBlock(n_channels, small_kernel_size) for _ in range(n_blocks)]
        self.block1 = nn.Sequential(*block1)

        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(n_channels)
        block2 = [_upsampleBlock(n_channels, 2) for _ in range(upsample_block_num)]
        self.block2 = nn.Sequential(*block2)

        self.conv3 = nn.Conv2d(n_channels, 3, kernel_size=large_kernel_size, padding=4)
        self._init_weights()

    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        block1 = self.block1(out1)
        out2 = self.bn(self.conv2(block1))
        block2 = self.block2(out1 + out2)
        out3 = self.conv3(block2)
        return torch.clamp_(out3, 0.0, 1.0)

    def _init_weights(self):
      for module in self.modules():
        if isinstance(module, nn.Conv2d):
          nn.init.kaiming_normal_(module.weight)
          if module.bias is not None:
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
          nn.init.constant_(module.weight, 1)

class _residualBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super(_residualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return x + out

class _upsampleBlock(nn.Module):
    def __init__(self, n_channels, up_scale):
        super(_upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        kernel_size = config.D.kernel_size
        n_channels = config.D.n_channels
        n_blocks = config.D.n_blocks
        fc_size = config.D.fc_size

        self.conv1 = nn.Conv2d(3, out_channels=n_channels, kernel_size=3, stride=1)
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        block = []
        for i in range(1, n_blocks//2, 1):
          block.append(_conv2dBlock(n_channels, n_channels, kernel_size, stride = 2))
          block.append(_conv2dBlock(n_channels, n_channels*2, kernel_size, 1))
          n_channels = n_channels*2
        block.append(_conv2dBlock(n_channels, n_channels, kernel_size, stride=2))
        self.block = nn.Sequential(*block)
      
        self.fc1 = nn.Linear(n_channels*6*6, fc_size)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(fc_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.leakyrelu1(self.conv1(x))
        out = self.block(out)
        out = torch.flatten(out, 1)
        out = self.leakyrelu2(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out
    
class _conv2dBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(_conv2dBlock, self).__init__()
      self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
      self.bn = nn.BatchNorm2d(out_channels)
      self.relu = nn.LeakyReLU(0.2)
  
  def forward(self, x):
      return self.relu(self.bn(self.conv(x)))


class TruncatedVGG19(nn.Module):
  def __init__(self, i, j):
    assert i > j 
    super(TruncatedVGG19, self).__init__()
    vgg = torchvision.models.vgg19(pretrained=True)
    if i<=2:
      self.model = nn.Sequential(*list(vgg.features)[:j*4+i])
    else:
      self.model = nn.Sequential(*list(vgg.features)[:2*4+(i-2)*8+i-1])

    for param in self.model.parameters():
        param.requires_grad = False

  def forward(self, imgs):
    return self.model(imgs)
