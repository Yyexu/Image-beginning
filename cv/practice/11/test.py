import h5py

# 替换成你的文件路径
with h5py.File('91-image_x2.h5', 'r') as f:
    print("训练集中的键:", list(f.keys()))

with h5py.File('Set5_x2.h5', 'r') as f:
    print("测试集中的键:", list(f.keys()))