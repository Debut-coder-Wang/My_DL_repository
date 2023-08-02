import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torchvision.utils import save_image

'''
这个算法的核心思想是通过将一个原始图像的内容和一个风格图像的风格进行结合，生成一张新的图像，使其既具有原始图像的内容特征，又具有风格图像的风格特征。
这种图像风格迁移的算法基于卷积神经网络，并通过最小化原始图像与生成图像之间的内容损失和风格损失来实现。

具体来说，算法通过预训练的VGG19模型提取原始图像、风格图像和生成图像的特征。
然后，使用特征之间的距离度量，如均方误差（MSE），计算原始图像和生成图像之间的内容损失，以及风格图像和生成图像之间的风格损失。
内容损失用于保留原始图像的内容特征，而风格损失用于捕捉风格图像的风格特征。
最后，通过调整超参数alpha和beta来平衡内容和风格在生成图像中的贡献，从而得到最终的生成图像。优化过程使用Adam优化器来更新生成图像，并根据总损失进行反向传播。

总的来说，该算法的核心思想是利用卷积神经网络提取图像的内容和风格特征，并通过最小化内容损失和风格损失来合成具有原始图像内容和风格图像风格的新图像。
这种方式可以实现图像间的风格迁移，使得生成图像既保留了原始图像的内容，又展现了风格图像的独特风格。
'''

class VGG19(nn.Module):
    def __init__(self):
        super(self, VGG19).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features


def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 使用cpu就把尺寸弄小一点，用GPU就可以弄大一点
image_size = 256
total_steps = 6000
lr = 0.001
# alpha是content超参数，beta是style超参数，这两个值决定图像风格迁移的程度
alpha = 1
beta = 0.01
loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ]
)

original_image = load_image("G_class.jpeg")
style_image = load_image("transform.png")
generated = original_image.clone().requires_grad_(True)
optimizer = optim.Adam([generated], lr=lr)
model = VGG19().to(device).eval()

for step in range(total_steps):
    original_features = model(original_image)
    style_features = model(style_image)
    generated_features = model(generated)
    style_loss = original_loss = 0

    for gen_features, ori_features, sty_features in zip(
        generated_features, original_features, style_features
    ):
        batch_size, channel, height, width = gen_features.shape
        original_loss += torch.mean((gen_features-ori_features)**2)  # 计算内容损失

        G = gen_features.view(channel, height*width).mm(gen_features.view(channel, height*width).t())
'''
这两句代码计算了生成图像的Gram矩阵G，用于衡量生成图像的风格。

首先，gen_features.view(channel, height*width)将生成图像的特征张量重新形状为一个2D矩阵，其中行数为通道数channel，列数为图像高度height乘以宽度width。
这样操作后，每一行代表一个通道的特征向量。

然后，.mm(gen_features.view(channel, height*width).t())进行了矩阵相乘操作，计算生成图像特征矩阵与其转置矩阵的乘积。
结果是一个Gram矩阵，大小为channel x channel，其中每个元素表示对应通道之间的相关性。

Gram矩阵的计算可以捕捉到特定图像的纹理和风格信息。通过测量不同通道之间的相关性，Gram矩阵能够反映出图像中不同尺度和层次上的纹理组合方式。
在图像风格迁移中，通过计算原始图像和风格图像的Gram矩阵，以及生成图像与风格图像的Gram矩阵之间的均方误差，来量化生成图像与风格图像的风格相似度。
'''
        A = sty_features.view(channel, height*width).mm(sty_features.view(channel, height*width).t())

        style_loss += torch.mean((G-A)**2)  # 使用均方误差计算风格损失

    total_loss = alpha*original_loss + beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        save_image(generated, "generated.png")
