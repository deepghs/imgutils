from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, avgpool_size: Tuple[int, int] = (4, 4)):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(avgpool_size)
        self.linear = nn.Linear(512 * block.expansion * avgpool_size[0] * avgpool_size[1], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        print(out.shape)
        out = self.avgpool(out)
        print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet182D(ResNet):
    __model_name__ = 'resnet18_2d'
    __dims__ = 2

    def __init__(self, num_classes: int = 2):
        ResNet.__init__(self, BasicBlock, [2, 2, 2, 2], num_classes)


class ResNet342D(ResNet):
    __model_name__ = 'resnet34_2d'
    __dims__ = 2

    def __init__(self, num_classes: int = 2):
        ResNet.__init__(self, BasicBlock, [3, 4, 6, 3], num_classes)


class ResNet502D(ResNet):
    __model_name__ = 'resnet50_2d'
    __dims__ = 2

    def __init__(self, num_classes: int = 2):
        ResNet.__init__(self, Bottleneck, [3, 4, 6, 3], num_classes)


class ResNet1012D(ResNet):
    __model_name__ = 'resnet101_2d'
    __dims__ = 2

    def __init__(self, num_classes: int = 2):
        ResNet.__init__(self, Bottleneck, [3, 4, 23, 3], num_classes)


class ResNet1522D(ResNet):
    __model_name__ = 'resnet152_2d'
    __dims__ = 2

    def __init__(self, num_classes: int = 2):
        ResNet.__init__(self, Bottleneck, [3, 8, 36, 3], num_classes)


if __name__ == '__main__':
    from thop import profile

    for resnet_class in [
        ResNet182D, ResNet342D,
        ResNet502D, ResNet1012D, ResNet1522D
    ]:
        net = resnet_class()
        x = torch.randn(1, 3, 384, 384)

        flops, params = profile(net, (x,))
        print(f'{resnet_class.__name__}:')
        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        print('Params = ' + str(params / 1000 ** 2) + 'M')
