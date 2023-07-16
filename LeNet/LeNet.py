import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

'''

1、这段代码首先实现了一个经典的卷积神经网络——LeNet，使用MNIST数据集进行训练
2、这段代码中还直观的展示了batch_size和lr对卷积神经网络训练效果的影响

实验设置：
将lr分别设置为0.1, 0.01, 0.001, 0.0001，然后遍历组合2, 64, 128，256 4种batch_size来训练
每个实验组合迭代10轮，通过tensorboard来记录实验结果

'''

learning_rate_list = [0.1, 0.01, 0.001, 0.0001]
batch_size_list = [2, 64, 128, 256]
epoch = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
class_num = 10


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
    transforms.Resize(size=(32, 32)),
    transforms.ToTensor()
])


class LeNet(nn.Module):
    def __init__(self, num):
        super(LeNet, self).__init__()
        self.ReLU = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.con1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=0)
        self.con2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=0)
        self.con3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=0)
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.linear2 = nn.Linear(in_features=84, out_features=num)

    def forward(self, x):
        x = self.ReLU(self.con1(x))
        x = self.pool(x)
        x = self.ReLU(self.con2(x))
        x = self.pool(x)
        x = self.ReLU(self.con3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.ReLU(self.linear1(x))
        x = self.linear2(x)
        return x


train_set = torchvision.datasets.MNIST(root="./data", train=True, transform=operation, download=True)
test_set = torchvision.datasets.MNIST(root="./data", train=False, transform=operation, download=True)

for batch_size in batch_size_list:
    for learning_rate in learning_rate_list:
        # 这里必须在每一个实验组合开始前初始模型，否则实验之间的参数会互相影响
        model = LeNet(num=class_num)
        print(model)
        model = model.to(device)
        writer = SummaryWriter(log_dir=f"log//Batch_Size={batch_size} Lr={learning_rate}")
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
        step = 0
        for i in range(epoch):
            if i == 0:
                print(f"初始测试集准确率：{check_acc(model=model, loader=test_loader) * 100:.2f}%")
                writer.add_scalar(tag="test_acc", scalar_value=check_acc(model=model, loader=test_loader),
                                  global_step=step)
            for image, target in tqdm(train_loader):
                image = image.to(device)
                target = target.to(device)
                predictions = model(image)
                loss = loss_fn(predictions, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            step += 1
            print(f"Batch_Size={batch_size} Lr={learning_rate}")
            print(f"第{i + 1}轮训练结束，测试集准确率计算中......")
            print(f"测试集准确率：{check_acc(model=model, loader=test_loader) * 100:.2f}%")
            print("日志记录中......")
            writer.add_scalar(tag="test_acc", scalar_value=check_acc(model=model, loader=test_loader), global_step=step)

'''
实验结果：
效果最差的都是学习率0.1的，根本没办法训练
效果好的几乎都是使用0.001的，batch是128或256的
'''
