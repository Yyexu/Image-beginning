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
    target_matrix = np.full((h,w,img.shape[2]), background ,dtype=np.uint8)

    yy, xx = np.indices((h, w))
    ones = np.ones((h,w))
    target_hom = np.stack([xx, yy, ones], axis=-1).reshape(-1,3)
    # 映射矩阵
    matrix_change = np.array([[1,0,-tx],[0,1,-ty],[0,0,1]])

    src_hom = target_hom @ matrix_change.T

    src_x = src_hom[:, 0].astype(int).reshape(h, w)
    src_y = src_hom[:, 1].astype(int).reshape(h, w)

    mask = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)

    # 合法范围的像素赋值 筛选符合
    target_matrix[mask] = img[src_y[mask], src_x[mask]]

    return target_matrix



# 2.缩放(已修改)
# target_h target_w 期望缩放成的高和宽像素
# mode (nearest、linear)两种插值方法

def resize(img, target_h, target_w, mode='nearest', background=[255,255,255]):
    # 获取原图的高和宽
    orig_h, orig_w = img.shape[:2]
    # 构造目标像素的画布（初始化为背景色）
    target_matrix = np.full((target_h, target_w, img.shape[2]), background, dtype=np.uint8)
    # 获取缩放比例（反映射） 比例 = 原来 / 目标
    h_scale = orig_h / target_h
    w_scale = orig_w / target_w

    # 分别生成0-(h-1) 和0-(w-1)的整数网格
    yy, xx = np.indices((target_h, target_w))
    ones = np.ones_like(xx)
    target_hom = np.stack([xx, yy, ones], axis=-1).reshape(-1, 3)

    # ---- 构造缩放变换矩阵（反映射） ----
    matrix_change = np.array([[w_scale, 0, 0],
                              [0, h_scale, 0],
                              [0, 0, 1]], dtype=np.float32)

    # ---- 一次性计算所有原图坐标 ---- 广播 hxw,3
    src_hom = target_hom @ matrix_change.T
    # 还原成hxw的坐标网格
    src_x = src_hom[:, 0].reshape(target_h, target_w)
    src_y = src_hom[:, 1].reshape(target_h, target_w)

    # -------------------- 最近邻插值 --------------------
    if mode == 'nearest':
        src_xn = np.round(src_x).astype(int)
        src_yn = np.round(src_y).astype(int)

        # 越界检测 2x2
        mask = (src_xn >= 0) & (src_xn < orig_w) & (src_yn >= 0) & (src_yn < orig_h)

        # 合法范围的像素赋值 筛选符合
        target_matrix[mask] = img[src_yn[mask], src_xn[mask]]

    # -------------------- 双线性插值 --------------------
    elif mode == 'linear':
        # 计算四邻点
        x0 = np.floor(src_x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(src_y).astype(int)
        y1 = y0 + 1

        # 越界检测（必须保证四点都在范围内）
        mask = (x0 >= 0) & (x1 < orig_w) & (y0 >= 0) & (y1 < orig_h)

        # 权重
        dx = src_x - x0
        dy = src_y - y0

        # 只对合法区域进行插值
        valid = np.where(mask)
        Ia = img[y0[valid], x0[valid]]
        Ib = img[y0[valid], x1[valid]]
        Ic = img[y1[valid], x0[valid]]
        Id = img[y1[valid], x1[valid]]

        wx = dx[valid][:, np.newaxis]
        wy = dy[valid][:, np.newaxis]

        # 双线性插值公式
        val = ((1 - wy) * ((1 - wx) * Ia + wx * Ib) +
               wy * ((1 - wx) * Ic + wx * Id)).astype(np.uint8)

        target_matrix[valid] = val

    return target_matrix


# 3.旋转(已修改)
# center 为旋转中心点的元组，默认为图像中心(None)
# scale 缩放倍率，和resize不同，直接写倍率这个是，默认为1
# angle 旋转角度，默认为逆时针旋转
# background 填补空白区域的颜色
# mode (nearest、linear)两种插值方法
def rotate_fast(img, center=None, scale=1, angle=0, background=[255,255,255], mode='nearest'):
    h, w = img.shape[:2]
    if center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = center

    # 构造目标坐标网格
    yy, xx = np.indices((h, w))
    ones = np.ones_like(xx)
    target_hom = np.stack([xx, yy, ones], axis=-1).reshape(-1, 3)

    # 旋转矩阵
    sin_a = np.sin(np.deg2rad(angle))
    cos_a = np.cos(np.deg2rad(angle))
    M = np.array([
        [scale * cos_a, -scale * sin_a, (1 - scale * cos_a) * cx + scale * sin_a * cy],
        [scale * sin_a,  scale * cos_a, (1 - scale * cos_a) * cy - scale * sin_a * cx],
        [0, 0, 1]
    ])
    M_inv = np.linalg.inv(M)

    # 一次性反映射
    src_hom = target_hom @ M_inv.T
    src_x = src_hom[:, 0].reshape(h, w)
    src_y = src_hom[:, 1].reshape(h, w)

    # 输出初始化为 background
    out = np.full_like(img, background, dtype=np.uint8)

    if mode == 'nearest':
        src_xn = np.round(src_x).astype(int)
        src_yn = np.round(src_y).astype(int)
        mask = (src_xn >= 0) & (src_xn < w) & (src_yn >= 0) & (src_yn < h)
        out[mask] = img[src_yn[mask], src_xn[mask]]

    elif mode == 'linear':
        # 计算四邻坐标
        x0 = np.floor(src_x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(src_y).astype(int)
        y1 = y0 + 1

        # 合法范围掩码 (避免拉伸)
        mask = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)

        # 权重
        dx = src_x - x0
        dy = src_y - y0

        # 只对合法区域做插值
        valid = np.where(mask)
        Ia = img[y0[valid], x0[valid]]
        Ib = img[y0[valid], x1[valid]]
        Ic = img[y1[valid], x0[valid]]
        Id = img[y1[valid], x1[valid]]

        wx = dx[valid][:, None]
        wy = dy[valid][:, None]

        interp = ((1 - wy) * ((1 - wx) * Ia + wx * Ib) +
                  wy * ((1 - wx) * Ic + wx * Id)).astype(np.uint8)

        out[valid] = interp

    return out



img = cv.imread("1.jpg")  # cv2读取为BGR格式（不影响平移逻辑）

tranlated_img = resize(img,750,400)
# 显示结果
cv.imshow("Original", img)
cv.imshow("Translated", tranlated_img)
cv.waitKey(0)
cv.destroyAllWindows()
