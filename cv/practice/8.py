# 基于区域生长的图像分割

import cv2
import numpy as np

# ---------------------- 全局变量定义 ----------------------
seed_point = (-1, -1)  # 存储种子点坐标 (x, y)
original_img = None  # 原始图像
gray_img = None  # 灰度图像
segmented_img = None  # 分割结果图像
is_seed_selected = False  # 标记是否已选择种子点
GRAY_THRESHOLD = 15  # 灰度差阈值（可根据图像调整）
NEIGHBOR_MODE = 4  # 邻域模式：4邻域(4) / 8邻域(8)


# ---------------------- 鼠标回调函数：选择种子点 ----------------------
def mouse_callback(event, x, y, flags, param):
    global seed_point, is_seed_selected, segmented_img

    # 左键点击时记录种子点
    if event == cv2.EVENT_LBUTTONDOWN:
        # 确保点击坐标在图像范围内
        if 0 <= x < gray_img.shape[1] and 0 <= y < gray_img.shape[0]:
            seed_point = (x, y)
            is_seed_selected = True
            print(f"已选择种子点：x={x}, y={y}")

            # 初始化分割图像（与原始图像同尺寸，初始为黑色）
            segmented_img = np.zeros_like(original_img)

            # 执行区域生长
            region_growing(gray_img, seed_point, segmented_img)

            # 显示分割结果
            cv2.imshow("Segmented Result", segmented_img)


# ---------------------- 区域生长核心函数 ----------------------
def region_growing(gray_img, seed, output_img):
    """
    单种子区域生长算法（BFS实现）
    :param gray_img: 输入灰度图像
    :param seed: 种子点坐标 (x, y)
    :param output_img: 输出分割图像（绘制结果）
    """
    # 图像尺寸
    h, w = gray_img.shape
    # 种子点灰度值
    seed_gray = gray_img[seed[1], seed[0]]
    # 已访问标记矩阵（避免重复处理）
    visited = np.zeros((h, w), dtype=bool)
    # 邻域坐标（4邻域/8邻域）
    if NEIGHBOR_MODE == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8邻域

    # 初始化队列（BFS核心），存入种子点坐标 (y, x)
    queue = []
    queue.append((seed[1], seed[0]))
    visited[seed[1], seed[0]] = True

    # 开始生长
    while queue:
        # 取出队列头部像素
        y, x = queue.pop(0)

        # 将当前像素标记为前景（原始图像颜色）
        output_img[y, x] = original_img[y, x]

        # 遍历邻域像素
        for dy, dx in neighbors:
            ny = y + dy
            nx = x + dx

            # 检查坐标是否在图像范围内，且未被访问
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                # 计算当前邻域像素与种子点的灰度差
                gray_diff = abs(int(gray_img[ny, nx]) - int(seed_gray))

                # 满足灰度相似性准则则加入队列
                if gray_diff <= GRAY_THRESHOLD:
                    visited[ny, nx] = True
                    queue.append((ny, nx))


# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    # 1. 读取图像（替换为你的图像路径）
    img_path = "5.jpg"  # 建议用灰度/低噪声图像测试
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"无法读取图像，请检查路径：{img_path}")

    # 2. 转换为灰度图像（区域生长基于灰度特征）
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # 3. 创建窗口并绑定鼠标回调
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Original Image", mouse_callback)

    # 4. 显示图像，等待鼠标点击选种子
    print("请在图像窗口中左键点击选择种子点，按 'q' 退出程序")
    while True:
        cv2.imshow("Original Image", original_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按q退出
            break

    # 5. 释放资源
    cv2.destroyAllWindows()