# 1.图像灰度化与通道分离

import cv2 as cv
import numpy as np

# 灰度图转化
def gray_pic(img):
    b, g, r = cv.split(img)
    gray_img = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    return gray_img
#提取  单通道显示
def green(img):
    green_pic = img.copy()
    green_pic[:,:,0] = 0
    green_pic[:,:,2] = 0
    return green_pic

def red(img):
    red_pic = img.copy()
    red_pic[:,:,0] = 0
    red_pic[:,:,1] = 0
    return red_pic

def blue(img):
    blue_pic = img.copy()
    blue_pic[:,:,1] = 0
    blue_pic[:,:,2] = 0
    return blue_pic


img = cv.imread('1.jpg')
gray_img = gray_pic(img)
green_img = green(img)
red_img = red(img)
blue_img = blue(img)

cv.imshow("gray_pic", gray_img)
cv.imshow("green_pic", green_img)
cv.imshow("red_pic", red_img)
cv.imshow("blue_pic", blue_img)
cv.waitKey(0)
cv.destroyAllWindows()

