import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from MF import MatrixFactorization as mf
from LTOCF import LTOCF

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

df = pd.read_csv('../data/train_dataset.csv')



X = df['user_id']
Y= df['item_id']

x_train, x_valid, y_train, y_valid = train_test_split(X,Y,train_size=0.9,test_size=0.1,random_state=42,shuffle=False)


# 假设 num_users 和 num_items 分别为用户和图书的总数
num_users = max(df['user_id']) + 1
num_items = max(df['item_id']) + 1

# 初始化模型
num_topics = 10
model = LTOCF(num_users, num_items, num_topics).to(device)

# 假设使用随机梯度下降优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 存储损失值用于绘图
train_losses = []
# 假设训练过程
def train_model(model, optimizer, x_train, y_train, epochs=5, batch_size=64):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for idx in tqdm(range(0, len(x_train), batch_size)):
            batch_x = x_train[idx:idx + batch_size].to(device)
            batch_y = y_train[idx:idx + batch_size].to(device)

            optimizer.zero_grad()
            scores = model(batch_x, batch_y)

            loss = torch.nn.functional.mse_loss(scores, torch.zeros_like(scores))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)  # 计算累积的损失值
        epoch_loss /= len(x_train)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    # 绘制损失图像
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 2), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# 训练模型
#train_model(model, optimizer, torch.tensor(x_train.values), torch.tensor(y_train.values), epochs=10, batch_size=64)


# 假设评估模型的函数
def evaluate_model(model, x_valid, y_valid):
    model.eval()
    with torch.no_grad():
        x_valid_tensor = torch.tensor(x_valid.values).to(device)
        y_valid_tensor = torch.tensor(y_valid.values).to(device)
        predict = model.predict(x_valid_tensor)
        # 假设使用 F1 值作为评估指标
        f1 = f1_score(y_valid.values, predict,average='weighted')
    return f1

#torch.save(model, '../models/LTOCF.pth')
model = torch.load('../models/LTOCF.pth')


# 评估模型
#f1 = evaluate_model(model, x_valid, y_valid)
#print(f'Validation F1 score: {f1}')

# 生成test的最终结果
target = pd.read_csv('../data/test_dataset.csv')
target_tensor=torch.tensor(target['user_id'].values).to(device)
res = model.predict(target_tensor)

# 将结果数据转换为DataFrame
final_df = pd.DataFrame({'user_id': [row.item() for row in target_tensor.cpu()], 'item_id': [row for row in res]})
# 将DataFrame保存到final.csv
final_df.to_csv('../data/final.csv', index=False)
