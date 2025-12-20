import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ---------------------- 1. 配置基础参数（和训练时一致） ----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10  # CIFAR-10共10个类别
# 类别名称（和训练时一致）
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# ---------------------- 2. 定义和训练时完全一致的模型结构 ----------------------
# 注意：必须和训练代码里的SimpleCNN完全一样（改一个层都加载失败）
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------- 3. 加载模型权重（关键步骤） ----------------------
# 初始化模型
model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
# 加载.pth权重文件（替换为你的模型路径，比如'cifar10_cnn.pth'）
model_path = "cifar10_cnn.pth"
# 加载权重（map_location确保CPU/GPU都能加载）
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
# 切换到推理模式（禁用Dropout/BatchNorm，避免结果不准）
model.eval()
print("✅ 模型加载成功，已进入推理模式")

# ---------------------- 4. 定义图像预处理（和训练时的test_transform一致） ----------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10是32x32，必须缩放到一致尺寸
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])


# ---------------------- 5. 推理单张图像（核心：预测类别） ----------------------
def predict_image(image_path):
    # 1. 加载图像（支持jpg/png等格式）
    img = Image.open(image_path).convert('RGB')  # 转为RGB（避免灰度图/透明通道问题）
    # 2. 预处理
    img_tensor = transform(img).unsqueeze(0)  # 增加batch维度（模型要求输入是[batch, C, H, W]）
    img_tensor = img_tensor.to(DEVICE)
    # 3. 模型推理（禁用梯度计算，加速+省内存）
    with torch.no_grad():
        outputs = model(img_tensor)  # 输出：[1, 10]（每个类别的得分）
        # 4. 解析结果：取得分最高的类别
        _, predicted_idx = torch.max(outputs, 1)  # 得到类别索引
        predicted_class = classes[predicted_idx.item()]  # 转为类别名称
        # 可选：输出每个类别的概率（Softmax转换）
        probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        class_prob = {classes[i]: round(probabilities[i] * 100, 2) for i in range(NUM_CLASSES)}

    # 5. 返回结果
    print(f"📌 预测结果：{predicted_class}")
    print(f"📊 各类别概率：{class_prob}")
    return predicted_class


# ---------------------- 6. 运行推理（替换为你的测试图像路径） ----------------------
if __name__ == "__main__":
    # 替换为你的图像路径（比如CIFAR-10的测试图、自己拍的图）
    test_image_path = "猫.jpg"
    predict_image(test_image_path)