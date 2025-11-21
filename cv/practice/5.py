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
def Canny(img,TL=20, TH=40):
    # 1. 先将图片进行高斯滤波平滑处理
    img = gaussian_blur(img)
    # 2. 然后算出图片的梯度和角度
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    G_x = ndimage.convolve(img.astype(np.float32), kx, mode='reflect')
    G_y = ndimage.convolve(img.astype(np.float32), ky, mode='reflect')
    G = np.sqrt(G_x ** 2 + G_y ** 2)
    # 转化弧度
    angle = np.abs(G_y) / (np.abs(G_x) + 1e-6)

    mask0 = angle < 0.4142  # tan(22.5°)
    mask90 = angle > 2.4142  # tan(67.5°)
    mask90 = (np.abs(G_x) < 1e-6) | (angle > 2.4142)

    mask45 = (angle >= 0.4142) & (angle <= 2.4142) & (G_x * G_y > 0)
    mask135 = (angle >= 0.4142) & (angle <= 2.4142) & (G_x * G_y < 0)

    Z = np.zeros_like(G)

    # 中心区域（避免越界）
    c = G[1:-1, 1:-1]

    # ---------------------------
    # 0° 水平方向：比较左右
    # ---------------------------
    m = mask0[1:-1, 1:-1]
    left = G[1:-1, :-2]
    right = G[1:-1, 2:]
    keep = (c >= left) & (c >= right) & m
    Z[1:-1, 1:-1][keep] = c[keep]

    # ---------------------------
    # 45° 斜方向：左下 & 右上
    # ---------------------------
    m = mask45[1:-1, 1:-1]
    right_up = G[:-2, 2:]
    left_down = G[2:, :-2]
    keep = (c >= right_up) & (c >= left_down) & m
    Z[1:-1, 1:-1][keep] = c[keep]

    # ---------------------------
    # 90° 垂直方向：上 & 下
    # ---------------------------
    m = mask90[1:-1, 1:-1]
    up = G[:-2, 1:-1]
    down = G[2:, 1:-1]
    keep = (c >= up) & (c >= down) & m
    Z[1:-1, 1:-1][keep] = c[keep]

    # ---------------------------
    # 135° 斜方向：左上 & 右下
    # ---------------------------
    m = mask135[1:-1, 1:-1]
    left_up = G[:-2, :-2]
    right_down = G[2:, 2:]
    keep = (c >= left_up) & (c >= right_down) & m
    Z[1:-1, 1:-1][keep] = c[keep]

    strong = (Z >= TH)
    weak = (Z >= TL) & (Z < TH)

    res = np.zeros_like(Z, dtype=np.uint8)
    res[strong] = 255
    res[weak] = 100

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

            if 0 <= ny < H and 0 <= nx < W and res[ny, nx] == 100:
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

img_canny = Canny(img)
cv.imshow('img_canny', img_canny)
cv.waitKey(0)
cv.destroyAllWindows()

