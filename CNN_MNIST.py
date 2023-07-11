import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

'''==============================这段代码实现了一个简单的卷积神经网络，数据集为MNIST============================='''
# 超参数设置
learning_rate = 1e-3
batch_size = 64
epoch = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 准备数据集
train_set = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(),
                                       download=True)
test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(),
                                      download=True)
train_data = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
test_data = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


# 模型保存和读取函数
def save_checkpoint(state, file_name="check_point.pth.tar"):
    torch.save(state, file_name)


def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])


# acc 计算函数
def compute_acc(model, dataloader):
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x.to(device=device)
            x.to(device=device)
            prediction = model(x)
            _, pred_id = prediction.max(1)
            correct_num = (y == pred_id).sum()
            sample_num = x.size(0)
            acc = correct_num / sample_num
    model.train()
    return acc


# 创建模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(10 * 7 * 7, 100)
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


model = CNN()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

for i in range(epoch):
    check_point = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}
    #load_checkpoint(torch.load("check_point.pth.tar"), model, optimizer)
    print("test_acc:{:2f}%".format(compute_acc(model, test_data) * 100))
    if i % 2 == 0:  # 每两轮保存一次,打印一次准确率
        print("=>模型保存中.......<=")
        save_checkpoint(check_point)
        print("=>模型已保存<=")
        print("train_acc:{:2f}%".format(compute_acc(model,train_data)*100))
        print("test_acc:{:2f}%".format(compute_acc(model,test_data)*100))
    for batch_id, (data, target) in enumerate(tqdm(train_data)):
        data.to(device=device)
        target.to(device=device)
        prediction = model(data)
        loss = loss_fn(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("train acc:{:2f}%".format(compute_acc(model, train_data) * 100))
print("test_acc:{:2f}%".format(compute_acc(model, test_data) * 100))

"""
1、CNN的性能确实比MLP的性能好很多，第一次跑10轮训练和测试都达到了100%的准确率（可能是偶然，不过也很强了）
"""
