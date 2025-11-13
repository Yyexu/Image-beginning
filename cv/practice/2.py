# 2.图像几何变换
# 平移、旋转、缩放、垂直偏移、水平偏移
#
import cv2 as cv
import numpy as np
# 1.平移
# tx,ty 水平和垂直平移量
# background 为图片外像素的颜色底色
# 缺点 ： 现在做的平移不超过原来的画布大小,opencv的可以设置画布大小
#        而且没有适配灰度图（做一个分支判断就行）
def translation(img,tx=0,ty=0,background=[255,255,255]):
    # 获取原画布的尺寸
    h,w = img.shape[:2]
    # 平移后 的画布大小不变
    target_matrix = np.zeros([h,w,img.shape[2]],dtype=np.uint8)
    # 映射矩阵
    Matrix_change = np.array([[1,0,-tx],[0,1,-ty],[0,0,1]])
    for target_y in range(h):
        for target_x in range(w):
            # 齐次坐标，目标点的
            target_hom = np.array([target_x, target_y, 1],dtype=np.int32)
            # 计算得到原来的图的坐标
            src_hom = np.matmul(Matrix_change,target_hom,dtype=np.int32)
            # 对应原来像素的坐标
            src_x = src_hom[0]
            src_y = src_hom[1]
            if(src_x <0 or src_y <0 or src_x >w-1 or src_y >h-1):
                target_matrix[target_y,target_x] = background
            else:
                target_matrix[target_y,target_x] = img[src_y,src_x]
    return target_matrix

# 2.缩放
def resize(img,target_h,target_w,mode):
    # 获取原图的高和宽
    orig_h, orig_w = img.shape[:2]
    # 构造目标像素的画布（无颜色）
    target_matrix = np.zeros((target_h,target_w,img.shape[2]),dtype=np.uint8)
    # 获取缩放比例（反映射） 比例 = 原来/目标
    h_scale = orig_h / target_h
    w_scale = orig_w / target_w
    # 构造变换矩阵的逆矩阵 3x3
    Matrix_change = np.array([[w_scale,0,0],[0,h_scale,0],[0,0,1]])
    # 循环目标画布的像素，因为是反映射
    for target_y in range(target_h):
        for target_x in range(target_w):
            # 目标的齐次坐标
            target_hom = np.array([target_x, target_y, 1], dtype=np.float32)
            # 计算原来的齐次坐标 B=AX
            src_hom = np.matmul(Matrix_change,target_hom,dtype=np.float32)
            # 计算原来的坐标，并采用不同的插值方法求整数像素值
            src_x = src_hom[0]
            src_y = src_hom[1]
            # 最近邻插值
            if mode == 'nearest':
                # round + 防越界
                src_x = np.clip(round(src_x), 0, orig_w - 1)
                src_y = np.clip(round(src_y), 0, orig_h - 1)
                target_matrix[target_y, target_x] = img[src_y, src_x]
            # 双线性插值
            elif mode == 'linear':
                # 求周围四点像素的坐标
                y0, y1 = int(np.floor(src_y)), int(np.ceil(src_y))
                x0, x1 = int(np.floor(src_x)), int(np.ceil(src_x))
                # 防越界
                y0 = np.clip(y0, 0, orig_h-1)
                y1 = np.clip(y1, 0, orig_h-1)
                x0 = np.clip(x0, 0, orig_w-1)
                x1 = np.clip(x1, 0, orig_w-1)
                # 计算权重 floor取下限，为正值（0-1）
                dy = src_y - y0
                dx = src_x - x0
                # 左下+右下+左上+右上
                val = (1 - dy) * (1 - dx) * img[y0, x0] + \
                      (1 - dy) * dx * img[y0, x1] + \
                      dy * (1 - dx) * img[y1, x0] + \
                      dy * dx * img[y1, x1]
                target_matrix[target_y, target_x] = val.astype(np.uint8)
    return target_matrix

# 3.旋转
#


img = cv.imread('1.jpg')
py_img = translation(img,-200,-200)
cv.imshow('py_img',py_img)
cv.waitKey(0)
