# Sobel 算子
# Prewitt 算子
# Laplacian 算子
# Canny 算子（包括非极大值抑制和双阈值连接）\
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy import *

# Prewitt算子
def Prewitt(img):
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    G_x = ndimage.convolve(img.astype(np.float32), kx, mode='reflect')
    G_y = ndimage.convolve(img.astype(np.float32), ky, mode='reflect')
    G = np.sqrt(G_x ** 2 + G_y ** 2)

    G_x = np.abs(G_x)
    G_y = np.abs(G_y)
    G = np.abs(G)
    # 归一化
    G_x = np.clip(G_x, 0, 255)
    G_y = np.clip(G_y, 0, 255)
    G = np.clip(G, 0, 255)

    return G_x.astype(np.uint8), G_y.astype(np.uint8), G.astype(np.uint8)

# Sobel算子算法
def Sobel(img):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    G_x = ndimage.convolve(img.astype(np.float32), kx, mode='reflect')
    G_y = ndimage.convolve(img.astype(np.float32), ky, mode='reflect')
    G = np.sqrt(G_x ** 2 + G_y ** 2)

    G_x = np.abs(G_x)
    G_y = np.abs(G_y)
    G = np.abs(G)
    # 归一化
    G_x = np.clip(G_x, 0, 255)
    G_y = np.clip(G_y, 0, 255)

    # 二值化可增强亮度
    # G = np.clip(G, 0, 255)

    return G_x.astype(np.uint8), G_y.astype(np.uint8), G.astype(np.uint8)

# Scharr算子
def Scharr(img):
    kx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
    ky = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.float32)

    G_x = ndimage.convolve(img.astype(np.float32), kx, mode='reflect')
    G_y = ndimage.convolve(img.astype(np.float32), ky, mode='reflect')
    G = np.sqrt(G_x ** 2 + G_y ** 2)

    G_x = np.abs(G_x)
    G_y = np.abs(G_y)
    G = np.abs(G)
    # 归一化
    G_x = np.clip(G_x, 0, 255)
    G_y = np.clip(G_y, 0, 255)
    G = np.clip(G, 0, 255)

    # 二值化可增强亮度
    # G = np.clip(G, 0, 255)

    return G_x.astype(np.uint8), G_y.astype(np.uint8), G.astype(np.uint8)

# Laplace算子
def Laplace(img,T=100,max_value=255):
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)

    lap = ndimage.convolve(img, kernel)
    lap = np.abs(lap)
    lap = np.clip(lap, 0, 255)

    return lap.astype(np.uint8)

# Canny算子
def Canny(img,TL=50, TH=100):
    # 先将图片进行高斯滤波平滑处理
    img = gaussian_blur(img)
    # 然后算出图片的梯度和角度 Sobel算子
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    G_x = ndimage.convolve(img.astype(np.float32), kx, mode='reflect')
    G_y = ndimage.convolve(img.astype(np.float32), ky, mode='reflect')
    G = np.sqrt(G_x ** 2 + G_y ** 2)
    # 转化弧度 1e-6是为了减去误差
    theta = np.arctan2(G_y, G_x)
    H, W = G.shape
    Z = np.zeros_like(G)

    # 转成角度制 (0~180)
    angle = np.rad2deg(theta)
    # 负读书加180度
    angle[angle < 0] += 180

    # 中心区域 避免图片边界
    c = G[1:-1, 1:-1]

    # ==========================
    #     方向分类（精确角度）
    # 0°, 45°, 90°, 135°
    # ==========================

    # 0°: -22.5° ~ 22.5°，157.5° ~ 180°
    mask0 = ((angle <= 22.5) | (angle >= 157.5))[1:-1, 1:-1]

    # 45°: 22.5° ~ 67.5°
    mask45 = ((angle > 22.5) & (angle <= 67.5))[1:-1, 1:-1]

    # 90°: 67.5° ~ 112.5°
    mask90 = ((angle > 67.5) & (angle <= 112.5))[1:-1, 1:-1]

    # 135°: 112.5° ~ 157.5°
    mask135 = ((angle > 112.5) & (angle <= 157.5))[1:-1, 1:-1]

    # ==========================
    #  方向 0° → 比较左右
    # ==========================
    left = G[1:-1, :-2]
    right = G[1:-1, 2:]
    keep = (c >= left) & (c >= right) & mask0
    Z[1:-1, 1:-1][keep] = c[keep]

    # ==========================
    #  方向 45° → 比较右上 & 左下
    # ==========================
    right_up = G[:-2, 2:]  # ↗
    left_down = G[2:, :-2]  # ↙
    keep = (c >= right_up) & (c >= left_down) & mask45
    Z[1:-1, 1:-1][keep] = c[keep]

    # ==========================
    #  方向 90° → 上 & 下
    # ==========================
    up = G[:-2, 1:-1]
    down = G[2:, 1:-1]
    keep = (c >= up) & (c >= down) & mask90
    Z[1:-1, 1:-1][keep] = c[keep]

    # ==========================
    #  方向 135° → 左上 & 右下
    # ==========================
    left_up = G[:-2, :-2]  # ↖
    right_down = G[2:, 2:]  # ↘
    keep = (c >= left_up) & (c >= right_down) & mask135
    Z[1:-1, 1:-1][keep] = c[keep]

    # 双阈值
    # 大于High的为强点
    strong = (Z >= TH)
    # 大于Low小于High的是弱点
    weak = (Z >= TL) & (Z < TH)
    # 结果 先默认全为0
    res = np.zeros_like(Z, dtype=np.uint8)
    res[strong] = 255
    res[weak] = 50

    # 滞后连接
    q = deque()
    H, W = res.shape
    # 把所有 strong 边缘放入队列
    strong_pos = np.column_stack(np.where(strong))
    for y, x in strong_pos:
        q.append((y, x))

    # BFS 只扩展 weak=100
    while q:
        y, x = q.popleft()

        for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1), (0, 1),
                       (1, -1), (1, 0), (1, 1)]:

            ny, nx = y + dy, x + dx

            if 0 <= ny < H and 0 <= nx < W and res[ny, nx] == 50:
                res[ny, nx] = 255  # promote weak → strong
                q.append((ny, nx))

    # 未连接的 weak 全部抹掉
    res[res != 255] = 0
    return res


def gaussian_kernel_1d(size=5, sigma=1.0):
    k = size // 2
    x = np.arange(-k, k+1, 1)

    kernel_1d = np.exp(-(x**2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()  # 归一化

    return kernel_1d.astype(np.float32)

def gaussian_blur(img, size=5, sigma=1.0):
    g1 = gaussian_kernel_1d(size, sigma)
    # 结果
    result = np.zeros_like(img, dtype=np.float32)
    # 横向卷积（1xN）h
    temp = ndimage.convolve1d(img[:, :], g1, axis=1, mode='reflect')
    # 纵向卷积（Nx1）w
    temp = ndimage.convolve1d(temp, g1, axis=0, mode='reflect')
    result[:, :] = temp
    # 不返回np.uint8，不然会影响Canny的精度
    return result


#####################################
# 读取照片
img = cv.imread('dm.png',cv.IMREAD_GRAYSCALE).astype(np.float32)

'''
sobel_x, sobel_y, sobel = Sobel(img)
scharr_x, scharr_y, scharr = Scharr(img)
prewitt_x, prewitt_y, prewitt = Prewitt(img)
Laplace = Laplace(img)
'''

img_canny2 = Canny(img,25,50)
cv.imshow('img_canny2', img_canny2)
cv.waitKey(0)
cv.destroyAllWindows()

