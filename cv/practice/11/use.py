import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


# --- 1. 必须定义和训练时一模一样的模型结构 ---
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def predict(model_path, image_path, scale=2):
    # --- 2. 加载模型权重 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 3. 图像预处理 ---
    # 读取图像并转为 YCbCr
    img = Image.open(image_path).convert('YCbCr')
    w, h = img.size

    # 将整张图放大到目标尺寸 (Bicubic)
    # 这一步是必须的，因为 SRCNN 输入和输出尺寸一致
    img = img.resize((w * scale, h * scale), resample=Image.BICUBIC)
    y, cb, cr = img.split()

    # 准备输入张量 (只处理 Y 通道)
    input_tensor = transforms.ToTensor()(y).view(1, 1, y.size[1], y.size[0]).to(device)

    # --- 4. 模型推理 ---
    with torch.no_grad():
        output = model(input_tensor).clamp(0.0, 1.0)

    # --- 5. 后处理与保存 ---
    # 将 Tensor 转回 PIL Image
    output = output.cpu().squeeze().numpy()
    output_y = Image.fromarray((output * 255.0).astype(np.uint8), mode='L')

    # 合并增强后的 Y 通道和原有的 Cb, Cr 通道
    result_img = Image.merge('YCbCr', [output_y, cb, cr]).convert('RGB')

    # 保存结果
    output_path = "result_sr.png"
    result_img.save(output_path)
    print(f"超分辨率重建完成！已保存至: {output_path}")
    result_img.show()


if __name__ == "__main__":
    # 使用示例
    predict(
        model_path='srcnn_final.pth',  # 你的权重文件
        image_path='images.jpg',  # 你想放大的图片
        scale=2  # 放大倍数（需与训练时一致）
    )