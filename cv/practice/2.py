# 2.图像几何变换
# 平移、旋转、缩放、垂直偏移、水平偏移
#
import cv2 as cv
import numpy as np

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


img = cv.imread('1.jpg')
resize_img = resize(img,300,500,'linear')
resize_img2 = resize(img,300,500,'nearest')
cv.imshow('resize_img',resize_img)
cv.imshow('resize_img2',resize_img2)
cv.waitKey(0)
