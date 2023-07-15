import albumentations as A
import matplotlib.pyplot as plt
import cv2

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
'''使用torchvision中的transforms方法可以进行数据增强，但相比之下速度较慢且不那么丰富
   但是在这段代码中我将尝试使用albumentations来进行数据增强
   albumentations是一个非常强大的数据增强库，可以对几乎所有的视觉任务进行数据增强
   它运行起来非常快，并且具备相当多的数据增强方法，YYDS！
'''
# P是进行对应操作的概率，例如P=0.5，表示有0.5的概率图片会被进行该处理，有0.5的概率跳过这个操作什么也不干
My_transforms = A.Compose([
    # A.Resize(height=1920, width=1080, p=0.5),  # 尺寸变换
    # A.RandomCrop(height=224, width=224, p=0.5),  # 随机剪裁
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),  # 颜色抖动，改变图像的亮度、对比度、饱和度和色调
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),  # RGB通道色值偏移
    A.OneOf([
        A.HorizontalFlip(p=0.5),  # 水平翻转
        A.Rotate(limit=30, p=0.5)  # 旋转，旋转角度限制在30度
    ], p=1)  # 这个组合语句的意思是列表中的两个操作至少会发生一个
])

# 将图片读入并将其转换到RGB空间
img = cv2.imread("G_class.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 这里转换之后是一个字典，所以还要索引一下才能把图片找出来
plt.subplot(3, 4, 1)
plt.imshow(img)
plt.title("原图")
times = 11
for i in range(times):
    transformed = My_transforms(image=img)
    transformed_img = transformed["image"]
    plt.subplot(3, 4, i + 2)
    plt.imshow(transformed_img)
    plt.title(f"{i + 1}")
plt.show()

'''
问题记录（2023.7.15）：
1、遇到了matplotlib绘图标题不能显示中文的问题，
解决方法是下载了simhei字体放到了front文件夹下并修改了matplotlibrc文件导入了该字体，
最重要的一步是清除matplotlib的缓存然后再次导入环境，查看缓存地址的命令：
import matplotlib as mpl
print(mpl.get_cachedir())
'''
