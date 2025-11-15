# 4.空间滤波器的实现与应用
import cv2 as cv
import numpy as np

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

def blur(img, size=3):
    h, w, c = img.shape
    tc = size // 2
    tc_img = np.pad(img.astype(np.float32), ((tc, tc), (tc, tc), (0, 0)), mode='constant')
    windows = np.lib.stride_tricks.sliding_window_view(tc_img, (size, size, c))
    # 正确轴：windows.shape == (h, w, 1, size, size, c)
    blured = np.mean(windows, axis=(3, 4)).astype(np.uint8)  # 平均空间维度
    blured = np.squeeze(blured, axis=2)
    return blured

def mid(img, size=3):
    h, w, c = img.shape
    tc = size // 2
    tc_img = np.pad(img.astype(np.float32), ((tc, tc), (tc, tc), (0, 0)), mode='constant')
    windows = np.lib.stride_tricks.sliding_window_view(tc_img, (size, size, c))
    # 正确轴：windows.shape == (h, w, 1, size, size, c)
    mid = np.median(windows, axis=(3, 4)).astype(np.uint8)  # 平均空间维度
    mid = np.squeeze(mid, axis=2)
    return mid
import numpy as np

import numpy as np

def gaussian_kernel(size=3, sigma=1.0):
    """生成高斯核"""
    ax = np.arange(-size//2 + 1., size//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.astype(np.float32)

def gaussian_blur(img, size=3, sigma=1.0):
    h, w, c = img.shape
    r = size // 2

    # padding
    pad_img = np.pad(img.astype(np.float32),
                     ((r, r), (r, r), (0, 0)),
                     mode='constant')

    # 生成窗口：shape = (h, w, size, size, c)
    windows = np.lib.stride_tricks.sliding_window_view(
        pad_img, (size, size, c)
    )  # (h, w, 1, size, size, c)
    windows = windows[:, :, 0]  # 去掉那个 1 → (h, w, size, size, c)

    # 加权
    kernel = gaussian_kernel(size, sigma)  # (size, size)
    weighted = windows * kernel[:, :, None]  # 广播 → (h, w, size, size, c)

    # 求和，得到最终 (h, w, c)
    blured = weighted.sum(axis=(2, 3))

    return blured.astype(np.uint8)


#############################################


img = cv.imread('1.jpg')

# 噪声参数（按需调整）
salt_prob, pepper_prob = 0.03, 0.03  # 椒盐噪声概率
gauss_var = 0.01  # 高斯噪声方差（越大噪声越强）

# 添加噪声
sp_noisy = add_salt_pepper_noise(img, salt_prob, pepper_prob)
gauss_noisy = add_gaussian_noise(img, var=gauss_var)

# 均值滤波器
jz_img = blur(sp_noisy,size=3)
# 中值滤波器
mid_img = mid(sp_noisy,size=3)
#高斯滤波器
gauss_img = gaussian_blur(img, size=5, sigma=1.0)
gasus_edit = np.hstack((gauss_noisy, gauss_img))


cv.imshow('img', gasus_edit)
cv.waitKey(0)