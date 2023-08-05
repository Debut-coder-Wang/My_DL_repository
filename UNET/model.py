import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# 构造卷积块
class Double_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Double_Conv, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    # 这里有两种看法，一种是只将两层卷积看成块，一种是在此基础上包含一个最大池化下采样
    # 为了能够连贯的存储特征用于下采样这里采用第一种思路

    def forward(self, x):
        return self.Conv(x)


class UNET(nn.Module):
    def __init__(self, in_channel, out_channel, channels=None):
        super(UNET, self).__init__()

        # 方便后期将网络做深
        if channels is None:
            channels = [64, 128, 256, 512]

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义最下面的瓶颈层
        self.BottleNeck = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1] * 2, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(channels[-1] * 2),
            nn.ReLU(),
            nn.Conv2d(channels[-1] * 2, channels[-1] * 2, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(channels[-1] * 2),
            nn.ReLU()
        )
        # 最后一层降维卷积
        self.last_Conv = nn.Conv2d(channels[0], out_channel, kernel_size=(1, 1), stride=(1, 1), padding=0)
        # 用来存储下采样过程中每个阶段的特征，用于上采样
        self.down_sample_features = []
        # 分别存储下采样和上采样的操作
        self.down_sample = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        # 遍历通道，存储下采样阶段的操作
        for channel in channels:
            self.down_sample.append(Double_Conv(in_channels=in_channel, out_channels=channel))
            in_channel = channel
        # 反向遍历通道，存储上采样阶段的操作
        for channel in reversed(channels):
            self.up_sample.append(
                nn.ConvTranspose2d(channel * 2, channel, kernel_size=(2, 2), stride=(2, 2)))
            self.up_sample.append(Double_Conv(channel * 2, channel))

    # 开始构建计算框架
    def forward(self, x):
        # 下采样计算，把self.down_sample中的操作依次取出来对x做处理，并保留对应阶段的特征
        for down_step in self.down_sample:
            x = down_step(x)
            self.down_sample_features.append(x)
            x = self.MaxPool(x)
        # 瓶颈层操作（最底部）
        x = self.BottleNeck(x)
        self.down_sample_features = self.down_sample_features[::-1]
        # 上采样阶段，在上采样进行的过程中把self.down_sample_features取出来resize后做拼接然后接卷积
        for idx in range(0, len(self.up_sample), 2):
            x = self.up_sample[idx](x)
            if x.shape != self.down_sample_features[idx // 2]:
                self.down_sample_features[idx // 2] = TF.resize(self.down_sample_features[idx // 2], x.shape[2:])
            x = torch.cat((x, self.down_sample_features[idx // 2]), dim=1)
            x = self.up_sample[idx + 1](x)
        x = self.last_Conv(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNET(in_channel=1, out_channel=2).to(device)
x = torch.randn(1, 1, 572, 572).to(device)
print(model(x).shape)
