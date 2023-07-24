import torch
import torch.nn as nn


# 创建残差块: 1x1 --> 3x3 --> 1x1
# 这里有两种残差块，一种是将通道数量变成原来的4倍的残差结构，一种是不改变通道数量的残差结构
class block(nn.Module):
    def __init__(self, in_channel, out_channel, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * 4)
        self.ReLU = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()
        x = self.ReLU(self.bn1(self.conv1(x)))
        x = self.ReLU(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x = x + identity
        x = self.ReLU(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, img_channel, num_classes):
        super(ResNet, self).__init__()
        # 这里定义的初始化输入通道是给残差块迭代使用的
        self.in_channel = 64
        # 下面的卷积1是ResNet开头的第一个普通的卷积，目的是快速将图片下采样到一个较小的尺寸，减少参数
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(block, layers[0], 64, 1)
        self.layer2 = self._make_layer(block, layers[1], 128, 2)
        self.layer3 = self._make_layer(block, layers[2], 256, 2)
        self.layer4 = self._make_layer(block, layers[3], 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    # ResNet的核心部分，构造残差结构
    def _make_layer(self, block, num_block, intermediate_channel, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channel != 4 * intermediate_channel:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,
                          intermediate_channel * 4,
                          kernel_size=(1, 1),
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(intermediate_channel * 4)
            )
        # 这里指明了参数，采用扩大通道数量的残差结构，在网络的开头将通道扩大至之前的4倍，相应的残差特征也要扩大到4倍之后做加法
        layers.append(block(self.in_channel, intermediate_channel, identity_downsample, stride))
        self.in_channel = intermediate_channel * 4
        # 剩余的部分采用普通的残差结构，即不改变通道数量，直接与初始特征相加
        for i in range(num_block - 1):
            layers.append(block(self.in_channel, intermediate_channel))  # 这里没有写的参数全部都用默认值
        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


BATCH_SIZE = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
net = ResNet50(img_channel=3, num_classes=1000)
net = net.to(device)
x = torch.randn(4, 3, 224, 224)
x = x.to(device)
y = net(x)
# y = y.to(device)
assert y.size() == torch.Size([BATCH_SIZE, 1000])
print(y.size())
