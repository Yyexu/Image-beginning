# 基于CNN的图像分类 深度学习
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 配置全局参数 ----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先用GPU
print(f"目前使用的是 {DEVICE}")
BATCH_SIZE = 64  # 批次大小
EPOCHS = 20  # 训练轮数
LEARNING_RATE = 1e-3  # 学习率
NUM_CLASSES = 10  # CIFAR-10共10个类别

if torch.cuda.is_available():
    # 开启CuDNN基准模式（针对固定尺寸输入的卷积加速）
    torch.backends.cudnn.benchmark = True
    # 显式开启CuDNN（默认已开启，显式声明更稳妥）
    torch.backends.cudnn.enabled = True

# ---------------------- 2. 数据增强与数据加载 ----------------------
# 训练集：数据增强（随机水平翻转、随机旋转、归一化）
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转（概率50%）
    transforms.RandomRotation(degrees=15),  # 随机旋转±15度
    transforms.ToTensor(),  # 转为Tensor（0-1归一化）
    transforms.Normalize(  # 基于CIFAR-10均值/方差标准化
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

# 测试集：仅归一化（不做数据增强，避免干扰评估）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

# 加载CIFAR-10数据集（自动下载到本地）
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=False, transform=train_transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=False, transform=test_transform
)

# 构建数据加载器（批量加载、打乱、多线程）
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# CIFAR-10类别名称（用于可视化）
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# ---------------------- 3. 搭建简单CNN模型 ----------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 卷积层+池化层（提取空间特征）
        self.features = nn.Sequential(
            # 第一层卷积：3通道→32通道，卷积核3x3， padding=1（保持尺寸）
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),  # 激活函数（引入非线性）
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化：32x32→16x16

            # 第二层卷积：32通道→64通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化：16x16→8x8

            # 第三层卷积：64通道→128通道
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化：8x8→4x4
        )

        # 全连接层（将特征映射为类别概率）
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平：128*4*4 → 2048维向量
            nn.Linear(128 * 4 * 4, 512),  # 全连接层1：2048→512
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout（防止过拟合）
            nn.Linear(512, num_classes),  # 全连接层2：512→10（对应10个类别）
            # 注：训练时用CrossEntropyLoss（内置Softmax），故此处不写Softmax
        )

    def forward(self, x):
        x = self.features(x)  # 卷积提取特征
        x = self.classifier(x)  # 全连接分类
        return x


# 初始化模型并移至GPU/CPU
model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
print("模型结构：\n", model)

# ---------------------- 4. 定义损失函数与优化器 ----------------------
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（含Softmax，适配分类任务）
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam优化器


# ---------------------- 5. 训练函数 ----------------------
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()  # 切换训练模式（启用Dropout/BatchNorm）
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # 数据移至设备
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 前向传播
        outputs = model(images)  # 模型输出：(batch_size, 10)
        loss = criterion(outputs, labels)  # 计算损失

        # 反向传播+参数更新
        optimizer.zero_grad()  # 清空梯度（避免累积）
        loss.backward()  # 反向传播求梯度
        optimizer.step()  # 更新参数

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  # 取概率最大的类别（等价于Softmax后取argmax）
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 打印批次信息
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {running_loss / (batch_idx + 1):.4f}, Train Acc: {100 * correct / total:.2f}%')

    # 本轮训练结束，返回平均损失和准确率
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    return train_loss, train_acc


# ---------------------- 6. 测试函数 ----------------------
def test(model, test_loader, criterion):
    model.eval()  # 切换测试模式（禁用Dropout/BatchNorm）
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算（加速+节省内存）
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算测试集损失和准确率
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n')
    return test_loss, test_acc


# ---------------------- 7. 开始训练与评估 ----------------------
train_losses = []
train_accs = []
test_losses = []
test_accs = []

print("\n开始训练...")
for epoch in range(EPOCHS):
    # 训练一轮
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
    # 测试一轮
    test_loss, test_acc = test(model, test_loader, criterion)

    # 记录指标
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

# ---------------------- 8. 可视化训练过程 ----------------------
plt.figure(figsize=(12, 4))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('train_result.png')
plt.show()

# ---------------------- 9. 保存模型 ----------------------
torch.save(model.state_dict(), 'cifar10_cnn.pth')
print("模型已保存为 cifar10_cnn.pth")


# ---------------------- 10. 可视化测试集预测结果 ----------------------
def visualize_predictions(model, test_loader, classes, num_images=5):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # 反归一化（方便可视化）
    inv_normalize = transforms.Normalize(
        mean=[-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010],
        std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010]
    )

    # 绘制结果
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        ax = plt.subplot(1, num_images, i + 1)
        img = inv_normalize(images[i].cpu())  # 反归一化
        img = np.transpose(img, (1, 2, 0))  # Tensor(C,H,W)→Numpy(H,W,C)
        img = np.clip(img, 0, 1)  # 限制像素值0-1

        plt.imshow(img)
        plt.title(f"True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}")
        plt.axis('off')
    plt.show()


# 可视化预测结果
visualize_predictions(model, test_loader, classes)