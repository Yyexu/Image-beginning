# 4.空间滤波器的实现与应用
import cv2 as cv
import numpy as np
from scipy import ndimage

def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """添加椒盐噪声"""
    noisy = image.copy()
    rows, cols = noisy.shape[:2]
    # 盐噪声（白色）
    salt_mask = np.random.random((rows, cols)) < salt_prob
    noisy[salt_mask] = 255
    # 椒噪声（黑色）
    pepper_mask = np.random.random((rows, cols)) < pepper_prob
    noisy[pepper_mask] = 0
    return noisy

def add_gaussian_noise(image, mean=0, var=0.005):
    """添加高斯噪声"""
    noisy = image.astype(np.float32) / 255.0
    sigma = np.sqrt(var)
    noise = np.random.normal(mean, sigma, image.shape)
    noisy += noise
    noisy = np.clip(noisy, 0, 1)
    return (noisy * 255).astype(np.uint8)

# 均值滤波算法
def blur(img, size=3):
    kernel = np.ones((size, size), np.float32) / (size ** 2)
    h, w, c = img.shape
    result = np.zeros_like(img, dtype=np.float32)
    for ch in range(c):
        result[:, :, ch] = ndimage.convolve(img[:, :, ch].astype(np.float32),kernel,mode='reflect')

    return result.astype(np.uint8)


# 中值滤波算法
def median_blur(img, size=3):
    h, w, c = img.shape
    result = np.zeros_like(img, dtype=np.float32)

    for ch in range(c):
        result[:, :, ch] = ndimage.median_filter(img[:, :, ch].astype(np.float32),size=size,mode='reflect')

    return result.astype(np.uint8)


# 高斯滤波算法
# 求解一维高斯核
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

    # 对每个通道分开处理
    for ch in range(img.shape[2]):
        # 横向卷积（1xN）h
        temp = ndimage.convolve1d(img[:, :, ch], g1, axis=1)
        # 纵向卷积（Nx1）w
        temp = ndimage.convolve1d(temp, g1, axis=0)

        result[:, :, ch] = temp

    return result.astype(np.uint8)
#############################################


img = cv.imread('1.jpg')

# 噪声参数（按需调整）
salt_prob, pepper_prob = 0.03, 0.03  # 椒盐噪声概率
gauss_var = 0.01  # 高斯噪声方差（越大噪声越强）

# 添加噪声
sp_noisy = add_salt_pepper_noise(img, salt_prob, pepper_prob)
gauss_noisy = add_gaussian_noise(img, var=gauss_var)

# 均值滤波器
jz_img = blur(gauss_noisy,size=3)
# 中值滤波器
median_img = median_blur(sp_noisy,size=3)
# 高斯滤波器
gaussian_img = gaussian_blur(gauss_noisy)

cv.imshow('jz_img', jz_img)
cv.imshow('median_img', median_img)
cv.imshow('gaussian_img', gaussian_img)
cv.waitKey(0)
cv.destroyAllWindows()
