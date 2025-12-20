# 10. 图像超分辨率重建
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import math


# 1. 定义 SRCNN 模型 (针对单通道 Y)
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


# 2. H5 数据集读取类
class H5Dataset(Dataset):
    def __init__(self, h5_file):
        super(H5Dataset, self).__init__()
        self.h5_file = h5_file

        # 1. 预先检查文件结构和样本总量
        with h5py.File(self.h5_file, 'r') as f:
            # 判断是 Group 嵌套模式还是 Dataset 数组模式
            self.is_group = isinstance(f['lr'], h5py.Group)
            if self.is_group:
                self.num_samples = len(f['lr'].keys())
            else:
                self.num_samples = len(f['lr'])

        # 初始文件句柄为空，在每个进程第一次调用 __getitem__ 时才真正打开
        self.file = None

    def __getitem__(self, idx):
        # 2. 只有在真正读取数据时才打开文件
        if self.file is None:
            # libver='latest', swmr=True 可以显著提升读取效率
            self.file = h5py.File(self.h5_file, 'r', libver='latest', swmr=True)

        # 3. 根据存储模式读取数据
        if self.is_group:
            # 如果是 Group 模式，按字符串索引读取
            str_idx = str(idx)
            data = self.file['lr'][str_idx][()].astype(np.float32)
            label = self.file['hr'][str_idx][()].astype(np.float32)
        else:
            # 如果是 Dataset 模式，按数字切片读取
            data = self.file['lr'][idx].astype(np.float32)
            label = self.file['hr'][idx].astype(np.float32)

        # 4. 维度修正：确保返回 PyTorch 要求的 (C, H, W)
        # 处理 (H, W) -> (1, H, W)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if label.ndim == 2:
            label = label[np.newaxis, ...]

        # 处理 (H, W, C) -> (C, H, W)
        if data.ndim == 3 and data.shape[2] <= 3:
            data = data.transpose(2, 0, 1)
            label = label.transpose(2, 0, 1)

        return torch.from_numpy(data), torch.from_numpy(label)

    def __len__(self):
        return self.num_samples


# 3.计算 PSNR
def calc_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 10 * math.log10(1.0 / mse.item())

# 4.计算SSIM
def calc_ssim(img1, img2, window_size=11, sigma=1.5):
    """
    计算 SSIM (结构相似性)
    输入 img1, img2 为 [B, C, H, W] 的 Tensor，范围 [0, 1]
    """
    channels = img1.size(1)

    # 创建高斯核
    def gaussian(size, sigma):
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channels, 1, window_size, window_size).to(img1.device)

    # 计算均值、方差、协方差
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# 5.实际训练
def train():
    # 使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    # 1. 加载数据
    train_dataset = H5Dataset('91-image_x2.h5')
    train_loader = DataLoader(dataset=train_dataset, batch_size=256, num_workers=4, shuffle=True)
    test_dataset = H5Dataset('Set5_x2.h5')
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    # 2. 模型与优化
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    # Adam 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("开始训练...")
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 验证指标
        model.eval()
        avg_psnr = 0
        avg_ssim = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs).clamp(0.0, 1.0)

                avg_psnr += calc_psnr(preds, labels)
                avg_ssim += calc_ssim(preds, labels)

        avg_psnr /= len(test_loader)
        avg_ssim /= len(test_loader)

        print(
            f"Epoch {epoch + 1} | Loss: {epoch_loss / len(train_loader):.6f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

    torch.save(model.state_dict(), "srcnn_final.pth")
    print("模型已保存为 srcnn_final.pth")


if __name__ == "__main__":
    train()








