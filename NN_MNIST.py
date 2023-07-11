import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

'''==============================这段代码实现了一个简单的神经网络，数据集为MNIST============================='''
# 超参数设置
learning_rate = 1e-3
batch_size = 64
epoch = 3
input_size = 784
output_size = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 准备数据集
train_set = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(),
                                       download=True)
test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(),
                                      download=True)
train_data = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
test_data = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

'''
学习记录：
1、F.ReLU和nn.ReLU的区别是：F.ReLU常用在forward中，nn.ReLU常用在模型定义中
2、模型中的超参数尽量在模型外进行定义，这样后期维护和修改起来会非常方便
3、在模型训练阶段开始前使用命令module.train()来告诉机器我在训练模型，在模型测试阶段是要在测试之前加入module.eval()来声明是在测试模型（会自动屏蔽掉dropout、batch_norm等推理阶段不需要的步骤）
4、在准确率计算部分，.max(1) 表示在第二个维度来找最大值，这个函数返回两个列表，第一个列表是每一行中按列查找的最大值，
第二个列表为这些最大值在对应行的索引。刚好这个数据集的标签类对应坐标索引，
所以只需要计算预测的索引值与真实值之间相同的数量就能得到预测对的数量
5、在最后一层使用了softmax层那么就不要使用CrossEntropy Loss，很可能造成梯度消失(实验对比查看下面的实验记录，确实是有差距)
6、在某次卷积之后使用BatchNorm时，尽量不要在该卷积层设置Bias，也就是将Bias=False
'''


# 模型定义，为了使模型能够应对任意维度的数据，这里使用了类属于句，输入输出维度在超参数定义部分进行定义
class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.l1 = nn.Linear(784, 50)
        self.l2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(p=0.5)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.dropout(F.relu(self.l1(x)))
        x = self.l2(x)
        return x


model = NN(input_size=input_size, output_size=output_size)
# 优化器和损失函数的定义
optim = optim.Adam(params=model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
# 训练阶段
for i in range(epoch):
    print("===============================第{}轮训练开始===============================".format(i+1))
    model.train()
    for batch_id, (data, target) in enumerate(tqdm(train_data)):
        data = data.to(device=device)
        target = target.to(device=device)
        data = data.reshape(data.shape[0], -1)
        pred = model(data)
        # 损失计算 ——> 梯度清零 ——> 反向传播 ——> 梯度更新
        loss = loss_fn(pred, target)
        optim.zero_grad()
        loss.backward()
        # 随机梯度剪裁，当使用LSTM、RNN等模型时为了预防梯度爆炸而进行的随机梯度剪裁
        #torch.nn.utils.clip_grad_norm(parameters=model.parameters(),max_norm=1)
        optim.step()


def compute_acc(model, data_loader):
    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            y_hat = model(x)
            _, idx = y_hat.max(1)
            correct_num = (idx == y).sum()
            sample_num = y_hat.size(0)
    model.train()  # 务必记住在使用  model.eval() 之后要将状态调整回训练状态
    return correct_num / sample_num

# 分别在训练数据和测试数据上使用训练完成的模型跑一遍准确率来观察过拟合现象
print(f"Accuracy on training set: {compute_acc(model,train_data)*100:.2f}")
print(f"Accuracy on test set: {compute_acc(model,test_data)*100:.2f}")
# print("test acc: {:.2f}%".format(compute_acc(model,test_data)*100))

'''
experiment log：
训练1轮：81.25%
训练10轮：92.18%
学习率调大一点之后训练10轮：96.87%
加dropout之后过你和现象明显减轻了
在使用softmax的前提下使用交叉熵损失：95.31%
不使用softmax：92.19%
使用梯度剪裁：90.62%
不使用梯度剪裁：95.31%（因为在这段代码中不怎么会造成梯度爆炸，所以剪裁梯度相反效果不怎么好）
'''
