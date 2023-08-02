import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torchvision.utils import save_image


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
    image = loader(image).unsqueese(0)
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
optimizer = optim.Adam(generated, lr=lr)
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
        original_loss += torch.mean((gen_features-ori_features)**2)

        G = gen_features.view(channel, height*width).mm(gen_features.view(channel, height*width).t())

        A = sty_features.view(channel, height*width).mm(sty_features.view(channel, height*width).t())

        style_loss += torch.mean((G-A)**2)

    total_loss = alpha*original_loss + beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        save_image(generated, "generated.png")
