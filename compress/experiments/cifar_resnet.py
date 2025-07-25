import torch
import torch.nn as nn
import torch.nn.functional as F


# uses option B for shortcut (seems to be the most common implementation)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.linear(out)


def resnet(depth, num_classes=10):
    assert (depth - 2) % 6 == 0, "Depth must be one of 20, 32, 44, 56, 110, 1202"
    n = (depth - 2) // 6
    return ResNet(BasicBlock, [n, n, n], num_classes)


def resnet_20(num_classes=10):
    return resnet(20, num_classes)


def resnet_56(num_classes=10):
    return resnet(56, num_classes)


def resnet_110(num_classes=10):
    return resnet(110, num_classes)


def resnet_1202(num_classes=10):
    return resnet(1202, num_classes)


def resnet20(num_classes=10):
    return resnet_20(num_classes)


def resnet56(num_classes=10):
    return resnet_56(num_classes)


def resnet110(num_classes=10):
    return resnet_110(num_classes)


def resnet1202(num_classes=10):
    return resnet_1202(num_classes)
