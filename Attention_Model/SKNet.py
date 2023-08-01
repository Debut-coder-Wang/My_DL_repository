import torch
import torch.nn as nn


'''
这段代码实现了SKNet中的注意力模块,可以调整超参数以实现不同的变体结构，具体细节请查看下方注释
'''

# in_channels表示输入特征的通道， M表示分支数， G表示groups分组卷积
# r表示Fuse阶段的降维维度， L表示最小的降维维度， 在 int(in_channels/r) 和 L 中选择大的那一个来作为降维参数
class SKAttention(nn.Module):
    def __init__(self, in_channels, M, G, r, L, stride=1):
        super(SKAttention, self).__init__()
        self.in_channels = in_channels
        self.M = M
        self.G = G
        self.d = max(int(in_channels / r), L)
        self.stride = stride
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Squeeze = nn.Sequential(
            nn.Conv2d(in_channels, self.d, kernel_size=(1, 1)),
            nn.ReLU(),
        )
        self.Excitation = nn.Sequential(
            nn.Conv2d(self.d, in_channels, kernel_size=(1, 1)),
            nn.Sigmoid()
        )
        self.split_conv = nn.ModuleList([])
        for i in range(M):
            self.split_conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=(3 + 2 * i, 3 + 2 * i),
                        stride=(stride, stride),
                        padding=1 + i,
                        groups=G
                    ),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU()
                )
            )

    def forward(self, x):
        features = []  # 用来存储不同感受野之下的特征，之后遍历它们使它们于之后的权重向量进行相乘
        feature_u = torch.zeros(x.size(), device=device)  # 这里生成的张量必须搬到device上，要不然会报设备不匹配的错
        for conv in self.split_conv:
            feature_u = torch.add(feature_u, conv(x))  # 首先融合split阶段的特征 feature_u
            features.append(conv(x))
        # 计算Fuse阶段的两个向量
        feature_s = self.avgpool(feature_u)
        feature_z = self.Squeeze(feature_s)
        for i in range(self.M):
            features[i] = features[i] * self.Excitation(feature_z)
        for feature in features:
            x = torch.add(x, feature)
        return x


# 检验，输入输出的张量维度应该相同
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
x = torch.randn(64, 64, 128, 128).to(device)
model = SKAttention(64, 3, 64, 4, 12).to(device)
print(model(x).shape)

"""
1、踩坑经历：不能只把最后的model和测试数据放到device上，代码中类似使用随机生成的变量等都要移到设备上才行，比如代码的43行
"""
