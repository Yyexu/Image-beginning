# Sobel 算子
# Prewitt 算子
# Laplacian 算子
# Canny 算子（包括非极大值抑制和双阈值连接）\
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
    # 二值阈值化处理
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)

    lap = ndimage.convolve(img, kernel)
    lap = np.abs(lap)
    lap = np.clip(lap, 0, 255)

    return lap.astype(np.uint8)
#####################################
# 读取照片
img = cv.imread('5.jpg',cv.IMREAD_GRAYSCALE).astype(np.float32)


sobel_x, sobel_y, sobel = Sobel(img)
scharr_x, scharr_y, scharr = Scharr(img)
prewitt_x, prewitt_y, prewitt = Prewitt(img)
Laplace = Laplace(img)

cv.imshow('Sobel', sobel)
cv.imshow('Scharr', scharr)
cv.imshow('Laplace', Laplace)

cv.waitKey(0)
cv.destroyAllWindows()

