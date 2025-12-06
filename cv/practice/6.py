# 7.图像的阈值处理
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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

    cv.imshow("75", guding1)
    cv.imshow("150", guding2)
    cv.imshow("225", guding3)
    cv.imshow("全局", qj)
    cv.imshow("OTSU", otsu_img)
    cv.waitKey(0)

