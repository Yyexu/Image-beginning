# 7.图像的阈值处理
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# 固定
def guding(img,thresh,maxval,minval):
    result = np.where(img>thresh,maxval,minval)
    return result.astype(np.uint8)
# 全局
def quanjv(img,maxval,minval):
    deleteT = 1
    hist = np.bincount(pic.flatten(), minlength=256)
    grayscale = np.arange(256)
    totalPixels = img.shape[0] * img.shape[1]
    totalGray = np.sum(grayscale * hist)
    T = int(np.round(totalGray / totalPixels))
    while True:
        numC1 = np.sum(hist[0:T])
        sumC1 = np.sum(hist[0:T] * np.arange(T))
        numC2 = totalPixels - numC1
        sumC2 = totalGray - sumC1
        T1 = int(np.round(sumC1 / numC1))
        T2 = int(np.round(sumC2 / numC2))
        Tnew = int(np.round((T1+T2)/2))
        print(f"全局阈值: {T}")
        if abs(T-Tnew) < deleteT:
            break
        else:
            T = Tnew

    return guding(img,T,maxval,minval)

# OTSU 但是还没有进行改进，使用向量改善for循环，但是向量化会增加一个数组，内存稍大一点
def otsu(img,maxval,minval):
    hist = np.bincount(img.flatten(), minlength=256)
    scale = np.arange(256)
    totalPixels = img.shape[0] * img.shape[1]
    totalGray = np.sum(scale * hist)
    mG = int(totalGray / totalPixels)
    icv = np.zeros(256)
    numFt,sumFt = 0,0
    for t in range(256):
        numFt += hist[t]
        sumFt += t * hist[t]
        pF = numFt / totalPixels
        mF = int((sumFt / totalPixels)) if numFt > 0 else 0

        numBt = totalPixels - numFt
        sumBt = totalGray - sumFt
        pB = numBt / totalPixels
        mB = int((sumBt / totalPixels)) if numBt > 0 else 0
        icv[t] = pF*(mF-mG)**2 + pB*(mB-mG)**2
    maxIndex = np.argmax(icv)
    print(f"OTSU阈值是: {maxIndex}")

    return guding(img, maxIndex, maxval, minval)

def compute_integral_image(image):
    H, W = image.shape
    integral = np.zeros((H + 1, W + 1), dtype=np.float32)

    for i in range(1, H + 1):
        for j in range(1, W + 1):
            integral[i, j] = (image[i - 1, j - 1] +  # 当前像素值
            integral[i - 1, j] +  # 上方积分值
            integral[i, j - 1] -  # 左侧积分值
            integral[i - 1, j - 1])  # 左上角积分值（重复相加部分）

        return integral

def compute_region_sum(integral, x1, y1, x2, y2):

    A = integral[x1, y1]  # 左上角
    B = integral[x1, y2 + 1]  # 右上角
    C = integral[x2 + 1, y1]  # 左下角
    D = integral[x2 + 1, y2 + 1]  # 右下角

    region_sum = D - B - C + A
    return region_sum

# 自适应阈值
def adaptive_threshold_mean(image,maxval,minval,block_size,C=0):

    H, W = image.shape
    output = np.zeros((H, W), dtype=np.uint8)

    # 1. 计算积分图
    integral = compute_integral_image(image)

    # 2. 计算每个像素的阈值
    half = block_size // 2

    for i in range(H):
        for j in range(W):
            # 计算邻域边界（考虑图像边界）
            x1 = max(0, i - half)
            x2 = min(H - 1, i + half)
            y1 = max(0, j - half)
            y2 = min(W - 1, j + half)

            # 使用积分图计算区域和
            region_sum = compute_region_sum(integral, x1, y1, x2, y2)

            # 计算区域面积
            area = (x2 - x1 + 1) * (y2 - y1 + 1)

            # 计算阈值（局部均值）
            threshold = region_sum / area - C

            # 应用阈值
            if image[i, j] > threshold:
                output[i, j] = 255

    return output

# 使用分离的高斯核加速
def adaptive_threshold_gaussian_ndimage(image, block_size, C=0, sigma=None):

    if sigma is None:
        sigma = block_size / 6

    # 使用 gaussian_filter 计算局部高斯加权均值
    # sigma 是标准差，truncate 控制核的大小（默认是 4.0，即 4*sigma）
    gaussian_mean = ndimage.gaussian_filter(image.astype(np.float32),
                                            sigma=sigma,
                                            mode='reflect')

    # 计算阈值
    thresholds = gaussian_mean - C

    # 应用阈值
    output = np.where(image > thresholds, 255, 0).astype(np.uint8)

    return output
if __name__ == '__main__':
    # 直接获得灰度图
    pic = cv.imread('1.jpg', cv.IMREAD_GRAYSCALE)
    # 获得直方图
    hist = np.bincount(pic.flatten(), minlength=256)

    # 固定阈值处理
    guding1 = guding(pic,75,255,0)
    guding2 = guding(pic,150,255,0)
    guding3 = guding(pic,225,255,0)

    # 全局阈值处理
    qj = quanjv(pic,255,0)

    # OTSU阈值算法
    otsu_img = otsu(pic,255,0)

    # 自适应阈值算法（局部均值和高斯）
    mean_thresh = adaptive_threshold_mean(otsu_img,250,0,3,0)
    guess_thresh = adaptive_threshold_gaussian_ndimage(otsu_img,5)

    cv.imshow("75", guding1)
    cv.imshow("150", guding2)
    cv.imshow("225", guding3)
    cv.imshow("全局", qj)
    cv.imshow("OTSU", otsu_img)
    cv.imshow("mean",mean_thresh)
    cv.imshow("guess",guess_thresh)

    cv.waitKey(0)

