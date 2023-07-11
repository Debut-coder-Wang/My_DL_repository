import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

'''================这段代码实现了一个简单的卷积神经网络，数据集为MNIST==============='''
# 超参数设置
learning_rate = 1e-3
batch_size = 64
epoch = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_model = True

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
if load_model:
    load_checkpoint(torch.load("check_point.pth.tar"), model, optimizer)
for i in range(epoch):
    if i == 0:
        print(f"initial test acc:{compute_acc(model,train_data)*100:.2f}%")
    if i % 2 == 0:  # 每两轮保存一次,打印一次准确率
        check_point = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}
        print("=>模型保存中<=")
        save_checkpoint(check_point)
        print("=>模型已保存<=")
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
2、保存的模型参数文件会被复写，所以始终只有一个权重文件
3、如果导入权重文件进行训练那么模型的初始化acc就会很高，因为继承了前几次训练好的参数，反之初始话acc就会很低（经过实验证明了这一点）
"""
