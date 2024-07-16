import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pandas as pd


# 数据集类
class GraphDataset(Dataset):
    def __init__(self, g, labels):
        self.g = g
        self.labels = labels

    def __getitem__(self, idx):
        return self.g[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


# GCN模型
class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hid_feats)
        self.conv2 = GraphConv(hid_feats, out_feats)

    def forward(self, g, features):
        h = F.relu(self.conv1(g, features))
        h = self.conv2(g, h)
        return h


# 图卷积层定义
class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, features):
        g.ndata['h'] = features
        g.update_all(fn.copy_u(u='h', out='m'), fn.sum(msg='m', out='h'))
        h = g.ndata['h']
        return self.linear(h)


# 训练函数
def train(model, g, features, labels, mask, optimizer, criterion, epochs):
    model.train()
    print(g)
    print(features)
    print(labels)
    print(mask)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(g, features)
        loss = criterion(output[mask], labels[mask])
        loss.backward()
        optimizer.step()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch + 1, loss.item()))


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask].cpu().numpy()
        labels = labels[mask].cpu().numpy()
        auc = roc_auc_score(labels, logits)
        return auc


def load_data():
    # 读取.cor文件
    df_cite = pd.read_csv('../data/cora/cora.cites', sep='\t', header=None, names=['source', 'target'])
    # 构建图
    # 将边的数据放入列表中
    src = df_cite['source'].tolist()
    dst = df_cite['target'].tolist()
    # 创建一个有向图
    g = dgl.graph((src, dst))
    # 读取特征和标签
    df_content = pd.read_csv('../data/cora/cora.content', sep='\t', header=None,
                             names=['paper_id'] + ['feat_' + str(i) for i in range(1435)] + ['label'])
    # 设置图的节点特征, 使用df的feat列
    features = df_content.iloc[:, 1:-1].values
    g.ndata['feat'] = features
    # 设置图的节点标签
    labels = df_content['label'].values
    g.ndata['label'] = labels
    # 将数据划分为训练集、验证集和测试集
    train_mask = df_content['label'].isin(['Neural_Networks', 'Genetic_Algorithms'])  # example train mask
    valid_mask = df_content['label'].isin(['Probabilistic_Methods'])  # example validation mask
    test_mask = df_content['label'].isin(['Case_Based'])  # example test mask
    # 在图中设置节点的划分标志
    g.ndata['train_mask'] = train_mask
    g.ndata['valid_mask'] = valid_mask
    g.ndata['test_mask'] = test_mask
    # 可选：将图转换为只包含训练节点的子图，加速训练
    train_idx = torch.nonzero(train_mask).squeeze()
    g_train = dgl.node_subgraph(g, train_idx)
    # 用于训练的数据
    train_features = g_train.ndata['feat']
    train_labels = g_train.ndata['label']
    return g, features, labels, train_mask, valid_mask, test_mask


def main():
    # 设置随机种子
    torch.manual_seed(0)
    np.random.seed(0)
    # 加载数据集
    dataset = 'citeseer'  # 可选'cora'或'citeseer'
    g, features, labels, train_mask, val_mask, test_mask = load_data()
    # 创建模型
    model = GCN(features.shape[1], 16, len(np.unique(labels)))
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 训练模型
    train(model, g, features, labels, train_mask, optimizer, criterion, epochs=100)
    train_auc = evaluate(model, g, features, labels, train_mask)
    val_auc = evaluate(model, g, features, labels, val_mask)
    test_auc = evaluate(model, g, features, labels, test_mask)
    print('Train AUC {:.4f} | Val AUC {:.4f} | Test AUC {:.4f}'.format(train_auc, val_auc,
                                                                       test_auc))


if __name__ == '__main__':
    main()
