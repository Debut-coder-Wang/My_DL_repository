import torch
import torch.nn as nn
from math import ceil  # 用来向上取整

# 这段代码首先实现基准模型即EfficientNetB0,其他几种衍生模型根据论文的描述只需要改变几个倍率因子来扩大网络即可
# 下面的参数分别是EfficientNetB0的：expand_ratio, channels, repeats, stride, kernel_size
base_model = [
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

# 这个字典里面装的是几种EfficientNet的超参数，分别是(Phi, resolution, drop_rate)
# 论文中的几个扩张因子：alpha, beta, gamma, 其中depth = alpha ** phi， width = beta**Phi, resolution = gama**Phi
# 论文中其实对分辨率操作的细节描述得不详细，下面的gama参数都是Google得来的，通过计算公式计算出分辨率：
# gama = 1.15, 对于b1,resolution = (gama**0.5)*224 = 240,其余的以此类推
phi_values = {
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


# 定义一个卷积块，可以减少代码篇幅；
# groups参数可以用来实现分组卷积，减小参数量， groups必须能整除in_channel 和 out_channel
class CNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.Block = nn.Sequential(
            # 这里的bias=False是因为下面使用了批量标准化，这是一个默认的操作
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.SiLU()
            # SiLU（）就是文中说的Swish激活函数：Swish = x*sigmoid(x),是谷歌提出来的一种性能良好的激活函数
        )

    def forward(self, x):
        return self.Block(x)


# SqueezeExcitation注意力模块，顾名思义就是先压缩然后拉长的操作
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dims):
        super(SqueezeExcitation, self).__init__()
        self.SE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 平均池化：CxHxW --> Cx1x1, 把特征变成一根的形状
            nn.Conv2d(in_channels, reduced_dims, kernel_size=(1, 1)),  # 使用1x1卷积降维
            nn.SiLU(),
            nn.Conv2d(reduced_dims, in_channels, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.SE(x)


# 下面构造的模块是EfficientNet中重复使用的MBConv和SE复合的卷积块，需要使用到SqueezeExcitation和CNNBlock
# expand_ratio是给MBConv用的,expand_ratio = {1,6}
# reduction是用来给SE模块指定降维用的
# survival_prob是用来实现stochastic depth的，随着网络的变深，有些卷积块其实是没用的，随机屏蔽掉一些卷积块，有效降低模型的计算负担
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio,
                 reduction=4, survival_prob=0.8):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1  # 满足后面两个条件时证明需要使用
        # MBConv6要对input进行升维，但是MBConv1不需要，所以下面将会根据MBConv的类型来进行判断返回哪种值
        self.hidden_dim = in_channels * expand_ratio
        self.ruduction_dim = int(in_channels / reduction)
        # 当in_channels != self.hidden_dim时表示使用的是MBConv6，此时表示需要升维卷积步骤
        self.expand = in_channels != self.hidden_dim
        # 创建升维卷积块
        if self.expand:
            self.expand_conv = CNNBlock(in_channel=in_channels, out_channel=self.hidden_dim, kernel_size=1, stride=1,
                                        padding=1)
        # 复合卷积块，不管是MBConv1还是MBConv6都要用到
        self.conv = nn.Sequential(
            CNNBlock(self.hidden_dim, self.hidden_dim, kernel_size, stride, padding, groups=self.hidden_dim),
            SqueezeExcitation(self.hidden_dim, self.ruduction_dim),
            nn.Conv2d(self.hidden_dim, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels)
        )

    # 构造随机深度函数，实现随机屏蔽一些残差块简化计算
    def stochastic_depth(self, x):
        if not self.training:
            return x
        else:
            # 按照survival_prob生成一个二值向量，小于survival_prob的为1，反之为0
            binary_tensor = (torch.randn(x.shape[0], 1, 1, 1, device=device) < self.survival_prob)
            # 第一步是在生成特征掩码，将此掩码乘以binary_tensor就可以随机使残差返回值变为特定值或者是0（屏蔽）
            return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, drop_rate = self.calculate_factors(version)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.last_channels = ceil(width_factor * 1280)
        self.conv_operations = self.creat_features(depth_factor, width_factor, self.last_channels)  # 所有的卷积操作都在这里面存储
        # 模型最后的全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.last_channels, num_classes),
        )

    # 找到对应版本的参数的函数
    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        Phi, resolution, drop_rate = phi_values[version]
        depth_factor = alpha ** Phi
        width_factor = beta ** Phi
        return width_factor, depth_factor, drop_rate

    def creat_features(self, depth_factor, width_factor, last_channels):
        out_channel = ceil(32 * width_factor)
        layers = []  # 用来存储一系列的卷积块最后交给nn.Sequential()函数解析
        # 添加第一层卷积，这是每个版本的EfficientNet共有的卷积层
        layers.append(CNNBlock(3, out_channel, 3, 2, 1))
        in_channels = out_channel
        # 开始遍历基准模型取参数来构建模型
        for expand_ratio, out_channels, repeats, stride, kernel_size in base_model:
            # 计算扩大后的输出通道，下面两个扩大输出通道的语句都可以用来扩大通道数量,但是有区别:
            # 第一句是直接找出离out_channels * width_factor最近的整数；第二句是找离out_channels * width_factor最近的且能被4整除的数
            # 做这个操作的原因是因为要做SE操作，SE中的reduction=4，即通道数要被压缩到原来的1/4，为了好除所以把通道变成4的倍数
            # out_channels = ceil(out_channels * width_factor)
            out_channels = 4 * ceil(int(out_channels * width_factor) / 4)
            # 计算扩大后的每个stage的深度
            repeat_times = ceil(repeats * depth_factor)
            for i in range(repeat_times):
                layers.append(InvertedResidualBlock(
                    in_channels, out_channels, kernel_size,
                    stride=stride if i == 0 else 1,  # 只在每个stage的第一层下采样
                    padding=kernel_size // 2,  # # if k=1:padding=0, k=3:padding=1, k=5:padding=2
                    expand_ratio=expand_ratio
                ))
                in_channels = out_channels  # 只在第一层改变通道，后面重复的部分不用改变通道都为out_channels
        # 最后一层卷积
        layers.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(self.conv_operations(x))
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
version = "b7"
_, resolution, _ = phi_values[version]
batch_size, num_classes = 4, 10
x = torch.randn((batch_size, 3, resolution, resolution)).to(device)
model = EfficientNet(version=version, num_classes=num_classes).to(device)
print(model(x).shape)
