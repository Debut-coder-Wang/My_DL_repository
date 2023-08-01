import torch
import torch.nn as nn

'''
这地代码实现了一个简单且高效的注意力模块--SqueezeExcitation模块（SE）
'''


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction_dims):
        super(SqueezeExcitation, self).__init__()
        # 整个SE模块的操作：
        self.SE_operations = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduction_dims, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(reduction_dims, in_channels, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.SE_operations(x)  # 返回的值应该是输入的特征乘以权重向量


device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.randn(4, 64, 128, 128).to(device)
SE = SqueezeExcitation(in_channels=64, reduction_dims=16).to(device)
print(SE(x).shape)

