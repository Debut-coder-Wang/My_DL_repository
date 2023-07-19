import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
epoch = 10
learning_rate = 1e-3
batch_size = 4
# 定义一个卷积操作，简化后面的代码
class ConvolutionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channel)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        return self.ReLU(self.batch_norm(self.conv(x)))


# 定义inception块，这是googleNet的精髓
class InceptionBlock(nn.Module):
    def __init__(self, in_chanel, out_channel1x1, REout_channel3x3, out_channel3x3, REout_channel5x5, out_channel5x5,
                 MaxPool_out_channel):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvolutionBlock(in_channel=in_chanel, out_channel=out_channel1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvolutionBlock(in_channel=in_chanel, out_channel=REout_channel3x3, kernel_size=1),
            ConvolutionBlock(in_channel=REout_channel3x3, out_channel=out_channel3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            ConvolutionBlock(in_channel=in_chanel, out_channel=REout_channel5x5, kernel_size=1),
            ConvolutionBlock(in_channel=REout_channel5x5, out_channel=out_channel5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvolutionBlock(in_channel=in_chanel, out_channel=MaxPool_out_channel, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


# 构造辅助分类器，防止梯度消失
# 这个模块只在训练的时候使用，推理的时候会被屏蔽掉
# 起到了类似正则化的作用，辅助分类器的值按照一个较小的权重加到最终的分类器值上

class AUX(nn.Module):
    def __init__(self, in_channel, class_num):
        super(AUX, self).__init__()
        self.conv = ConvolutionBlock(in_channel, 128, kernel_size=(1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=class_num)
        self.dropout = nn.Dropout(p=0.7)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 开始构建GoogleNet

class GoogleNet(nn.Module):
    def __init__(self, in_channel, AUX_permit, num_class):
        super(GoogleNet, self).__init__()
        # 进入网络后先通过两个普通的卷积和两个下采样（最大池化）
        self.conv1 = ConvolutionBlock(in_channel, out_channel=64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvolutionBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout2d(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=num_class)
        # 如果需要辅助分类器那么在实列化时AUX_permit将被设置为True
        if AUX_permit:
            self.aux1 = AUX(512, num_class)
            self.aux2 = AUX(528, num_class)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)  # 这里没有加激活是因为我们在上面创建的卷积块中已经加入了激活函数
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        x = self.inception4a(x)
        if AUX and self.training:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if AUX and self.training:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        if AUX and self.training:
            return x*+0.3*aux1+0.3*aux2  # 这里的返回值是将两个辅助分类器的输出以一个较小的权重加到输出x上，缓解网络过深梯度消失的现象
        else:
            return x

# 这里我把类别设置成了10，因为用的CIFAR10数据集，使用operation将size变成224x224，因为资源有限并且我只想验证模型是否能运行，所以没有下载imagenet这种巨大的数据集
model = GoogleNet(in_channel=3, AUX_permit=True, num_class=10).to(device)  
#print(model)


def check_acc(model, loader):
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            _, idx = prediction.max(1)
            correct_num = (idx == y).sum()
            sample = y.size(0)
    model.train()
    return correct_num / sample


operation = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="../data", train=True, transform=operation, download=True)
test_set = torchvision.datasets.CIFAR10(root="../data", train=False, transform=operation, download=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

for i in range(epoch):
    if i == 0:
        print(f"初始测试集准确率：{check_acc(model=model, loader=test_loader) * 100:.2f}%")
    for image, target in tqdm(train_loader):
        image = image.to(device)
        target = target.to(device)
        predictions = model(image)
        loss = loss_fn(predictions, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"第{i + 1}轮训练结束，测试集准确率计算中......")
    print(f"测试集准确率：{check_acc(model=model, loader=test_loader) * 100:.2f}%")

'''
1、第一次训练把辅助分类器返回值那里的加法写成了乘法（梯度消失更加严重了）
反映出来的现象就是无论怎么训练acc都是0，只返回最终的分类器值到预测阶段还是0
可见GoogleNet的辅助分类器加上去是有原因的，改正之后模型能够正确训练了
2、相比于VGG，GoogleNet的参数量确实下降了不少，同样的实验参数，VGG占了13.5G的现存，GoogleNet只占了8G不到
所以验证了inception模块降低参数的有效性（我认为主要还是1x1卷积起了重要作用）
3、启发：我认为inception这种思想可以使用到CV模型的方方面面，比较要想落地不能一味的把网络做深去刷acc
'''
