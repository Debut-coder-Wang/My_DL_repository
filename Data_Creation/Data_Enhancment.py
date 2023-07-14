from DataCreation import MyData
import torchvision.transforms as transforms
from torchvision.utils import save_image

operation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # 随机改变图像的亮度、对比度、饱和度和色调
    transforms.ToTensor()
])

dataset = MyData(label_dir="cats_dogs.csv", img_dir="cats_dogs_resized", transform=operation)

idx = 0
for i in range(2):  # 扩充数据集为原来的两倍，在这个过程中每张图片会按照上面的操作进行对应概率的转换
    for img, label in dataset:
        save_image(img,'img_' + str(idx) + '.png')
        idx += 1
