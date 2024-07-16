import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class LTOCF(torch.nn.Module):
    def __init__(self, num_users, num_items, num_topics, embedding_dim=64, dropout_p=0.5):
        super(LTOCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_topics = num_topics
        self.dropout_p = dropout_p

        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)
        self.topic_embedding = torch.nn.Embedding(num_topics, embedding_dim)

        # 主题矩阵 Phi
        self.topic_matrix = torch.nn.Parameter(torch.randn(embedding_dim, embedding_dim))

        # 残差连接的全连接层
        self.residual_layer = torch.nn.Linear(embedding_dim, embedding_dim)

        # Dropout 层
        self.dropout = torch.nn.Dropout(p=dropout_p)

        # Batch Normalization 层
        self.batch_norm = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)

        # 从物品嵌入中获取主题向量
        topic_vector = torch.matmul(item_embedding, self.topic_matrix)

        # 将主题向量整合到用户-物品交互中
        interaction = user_embedding * (item_embedding + topic_vector)

        # 添加残差连接
        residual = self.residual_layer(interaction)
        interaction = interaction + residual

        # Dropout 和标准化
        interaction = self.dropout(interaction)
        interaction = self.batch_norm(interaction)

        scores = torch.sum(interaction, dim=1)

        return scores

    def predict(self, user_ids, batch_size=1024):
        user_ids_tensor = torch.tensor(user_ids, dtype=torch.long).to(device)
        user_embedding = self.user_embedding(user_ids_tensor)

        item_ids = torch.arange(self.num_items).to(device)
        item_embedding = self.item_embedding(item_ids)
        topic_vector = torch.matmul(item_embedding, self.topic_matrix)

        num_items = item_embedding.size(0)
        num_batches = (num_items + batch_size - 1) // batch_size

        all_indices = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_items)

            batch_item_embedding = item_embedding[start_idx:end_idx]
            batch_interaction = user_embedding.unsqueeze(1) * (batch_item_embedding + topic_vector[start_idx:end_idx])

            # 添加残差连接
            batch_residual = self.residual_layer(batch_interaction)
            batch_interaction = batch_interaction + batch_residual

            # Dropout 和标准化
            batch_interaction = self.dropout(batch_interaction)
            batch_interaction = self.batch_norm(batch_interaction)

            batch_scores = torch.matmul(batch_interaction, item_embedding.transpose(0, 1))

            # 预测最高分项的索引
            _, batch_indices = torch.max(batch_scores, dim=1)
            all_indices.append(batch_indices.cpu().numpy())

        all_indices = np.concatenate(all_indices)

        return all_indices

