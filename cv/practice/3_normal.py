import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定微软雅黑为中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常的问题（可选，避免后续报错）


img = cv.imread('1.jpg')
# 1.转换为灰度图像
img_edit = (img[:,:,0]*0.587 + img[:,:,1]*0.114 + img[:,:,2]*0.299).astype(np.uint8)
# 2.统计得到直方图数组
hist = np.bincount(img_edit.flatten(), minlength=256)
# 3.获得直方图的CDF 概率分布函数
cdf = np.cumsum(hist)
# 4.CDF的线性缩放，目的是方便对比
cdf_normalized = cdf * float(hist.max()) / cdf.max()
# 5.搭建新老灰度的查找表，能加快速度
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
# 6.得到处理后的图
img_edit_2 = cdf[img_edit]
# 同理，做新图的直方图
hist2 = np.bincount(img_edit_2.flatten(), minlength=256)
cdf2 = np.cumsum(hist2)
cdf2_normalized = cdf2 * float(hist.max()) / cdf2.max()

plt.plot(cdf_normalized, color='blue', linewidth=2, label='原始CDF（归一化）')  # 标签1
plt.plot(cdf2_normalized, color='red', linewidth=2, label='均衡化后CDF（归一化）')  # 标签2
plt.legend()
plt.show()



