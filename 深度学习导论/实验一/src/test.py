from torch.utils.data import Dataset, random_split
import random
import torch
import numpy as np
import pandas as pd

# 生成数据
N = 10000
x = []
y = []
for i in range(N):
    x.append(random.uniform(1, 16))
    y.append(np.log2(x[-1]) + np.cos(np.pi * x[-1] / 2))

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)


# 定义数据集
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


# 创建数据集
dataset = MyDataset(x, y)

# 计算训练集、验证集和测试集的大小
num_total = len(dataset)
num_train = int(num_total * 0.8)
num_valid = int(num_total * 0.1)
num_test = num_total - num_train - num_valid

# 划分数据集
train_set, valid_set, test_set = random_split(dataset, [num_train, num_valid, num_test])

# 将torch.Tensor转换为numpy.ndarray
x_train_np = train_set.dataset.x[train_set.indices].numpy()
y_train_np = train_set.dataset.y[train_set.indices].numpy()
x_valid_np = valid_set.dataset.x[valid_set.indices].numpy()
y_valid_np = valid_set.dataset.y[valid_set.indices].numpy()
x_test_np = test_set.dataset.x[test_set.indices].numpy()
y_test_np = test_set.dataset.y[test_set.indices].numpy()

# 创建pandas.DataFrame
train_df = pd.DataFrame({'x': x_train_np, 'y': y_train_np})
valid_df = pd.DataFrame({'x': x_valid_np, 'y': y_valid_np})
test_df = pd.DataFrame({'x': x_test_np, 'y': y_test_np})

# 将数据集存储为CSV文件
train_df.to_csv('data/train.csv', index=False)
valid_df.to_csv('data/valid.csv', index=False)
test_df.to_csv('data/test.csv', index=False)