import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

# 2. 图片预处理设置
imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])


def image_loader(image_name):
    # 这里加个 try-except 防止找不到图片直接报错
    try:
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)
        return image.to(device)
    except FileNotFoundError:
        print(f"错误：找不到文件 {image_name}，请检查路径。")
        exit()


# --- 加载你的图片 ---
style_img = image_loader("fg.jpg")
content_img = image_loader("猫.jpg")

print("图片加载完成！")


# 3. 定义 Loss 和 Gram Matrix
class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = torch.nn.functional.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = torch.nn.functional.mse_loss(G, self.target)
        return input


# 4. 加载 VGG19
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# 5. 构建模型并提取特征
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, style_img, content_img):
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential()

    # 这里的 trick 是加上标准化层，但这示例里简化了，直接处理层
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers_default:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers_default:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
        model = model[:i + 1]

    return model, style_losses, content_losses


# ================== 【这里是修改的关键部分】 ==================

# 6. 初始化输入图片
input_img = content_img.clone()

# 【关键修改 1】: 必须先调用函数，获取包含 Loss 层的 model 和 loss 列表
print("正在构建风格迁移模型...")
model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)

# 优化器
optimizer = optim.LBFGS([input_img.requires_grad_()])

print('开始风格迁移训练...')
run = [0]
while run[0] <= 300:
    def closure():
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()

        # 【关键修改 2】: 这里必须用 model(input_img)，而不是 cnn(input_img)
        # 因为 model 里才插在这个 Loss 层，跑一遍 model 才会更新 Loss 值
        model(input_img)

        style_score = 0
        content_score = 0

        for sl in style_losses: style_score += sl.loss
        for cl in content_losses: content_score += cl.loss

        style_score *= 1000000
        content_score *= 1

        loss = style_score + content_score
        loss.backward()

        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Run {run[0]}: Style Loss : {style_score.item():.4f} Content Loss: {content_score.item():.4f}")
        return loss


    optimizer.step(closure)

# 7. 保存结果
input_img.data.clamp_(0, 1)
save_image(input_img, "output.jpg")
print("完成！结果已保存为 output.jpg")