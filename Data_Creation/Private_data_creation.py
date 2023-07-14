import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from DataCreation import MyData  # 这里要导入创建数据集的包

'''
使用torchvision中自带的dataset方法可以便捷的导入常见的数据集
但当我们使用自己的数据集时常常需要自己重写一下dataset
这段代码将实现使用自己的数据集训练模型
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epoch = 3


def check_acc(model, loader):
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x.to(device)
            y.to(device)
            preds = model(x)
            _, idx = preds.max(1)
            correct_num = (idx == y).sum()
            sample_num = preds.size(0)
        model.train()
    return correct_num / sample_num


# 创建自己的数据集
dataset = MyData(label_dir="cats_dogs.csv", img_dir="cats_dogs_resized", transform=transforms.ToTensor())
train_data, test_data = torch.utils.data.random_split(dataset, [5, 5])
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True, drop_last=True)

model = torchvision.models.googlenet(pretrained=False)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(in_features=1024, out_features=2)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3,weight_decay=1e-5)

for i in range(epoch):
    if i == 0:
        print(f"初始测试集准确率：{check_acc(model,test_loader)*100:.2f}%")
    for image,target in train_loader:
        image.to(device)
        target.to(device)
        prediction = model(image)
        prediction = prediction.logits # 使用GoogleNet需要做一下转换才能使用交叉熵损失函数做损失计算
        loss = loss_fn(prediction,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print(f"最终准确率：{check_acc(model,test_loader)*100:.2f}%")

