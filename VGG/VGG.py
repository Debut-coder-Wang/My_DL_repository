import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

learning_rate = 0.001
batch_size = 16
epoch = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
class_num = 10

architecture = {
    "VGG11": [64, 'M', 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, 'M', 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, 'M', 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, 'M', 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}


class VGG(nn.Module):
    def __init__(self, in_channels=3, class_num=10):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.Conv = self.get_architecture(architecture["VGG16"])
        self.fc = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=class_num)
        )

    def forward(self, x):
        x = self.Conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def get_architecture(self, architecture_list):
        operation = []
        in_channel = self.in_channels
        for x in architecture_list:
            if type(x) == int:
                out_channel = x
                operation += [
                    nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3), stride=(1, 1),
                              padding=1),
                    nn.BatchNorm2d(num_features=x), nn.ReLU()]
                in_channel = x
            else:
                operation += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*operation)


model = VGG(in_channels=3, class_num=10).to(device)
print(model)


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


"""
实验记录
1、模型确实大，笔记本狂啸，但真的有必要两个4096的全连接层吗，恐怖如斯
2、将batch_size设置成64，显存占有率大概13G，玩深度学习的门槛......,等到后期尝试一下减枝算法，hope it work!
"""
