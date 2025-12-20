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
hist_gram,bins = np.histogram(img_edit.flatten(), bins=256)
# 3.获得直方图的CDF 概率分布函数
cdf = np.cumsum(hist)
plt.figure()
plt.plot(cdf,label='cdf')
# 4.CDF的线性缩放，目的是方便对比
cdf_normalized = cdf * float(hist.max()) / cdf.max()
# 5.搭建新老灰度的查找表，能加快速度
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
# 6.得到处理后的图
img_edit_2 = cdf[img_edit]

hist2 = np.bincount(img_edit_2.flatten(), minlength=256)
cdf2 = np.cumsum(hist2)
plt.plot(cdf2,label='cdf2')
plt.show()


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.bar(range(256), hist)
plt.title("Histogram 1")

plt.subplot(1, 2, 2)
plt.bar(range(256), hist2)
plt.title("Histogram 2")

plt.tight_layout()
plt.show()