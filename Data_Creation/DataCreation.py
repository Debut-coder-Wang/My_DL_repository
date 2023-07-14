import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch


class MyData(Dataset):
    def __init__(self, label_dir, img_dir, transform):
        self.label = pd.read_csv(label_dir)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.label.iloc[index, 0])
        img = Image.open(img_path)
        label = torch.tensor(int(self.label.iloc[index, 1]))

        if self.transform:
            img = self.transform(img)

        return (img, label)
# class MyData(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.annotations = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.annotations)
#
#     def __getitem__(self, index):
#         img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
#         image = Image.open(img_path)
#         y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
#
#         if self.transform:
#             image = self.transform(image)
#
#         return (image, y_label)


'''
1、这里注意datasets和Dataset,前者是一个方法属于torchvision，后者是一个类属于torch  
'''
