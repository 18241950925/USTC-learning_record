import itertools

import dgl
import dgl.function as fn
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import citation_graph as citegrh
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset

i = 0


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
    def __init__(self, in_feats, hid_feats, out_feats, dropedge_prob=0.5):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hid_feats)
        self.conv2 = GraphConv(hid_feats, out_feats)
        self.dropedge_prob = dropedge_prob

    def forward(self, g, features):
        # g,features = self.dropedge(g, features)
        h = F.sigmoid(self.conv1(g, features))
        h = self.conv2(g, h)
        return h

    def dropedge(self, g, features):
        if self.training and self.dropedge_prob > 0:
            g = g.clone()  # Clone the graph to avoid modifying the original graph
            device = g.device
            num_edges = g.num_edges()
            mask = torch.rand(num_edges).to(device) > self.dropedge_prob
            g = g.edge_subgraph(mask)
            h = features[g.ndata['_ID']]  # 只选择对应的节点特征
            return g, h
        else:
            return g, features


class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, features):
        g.ndata['h'] = features
        g.update_all(fn.copy_u(u='h', out='m'), fn.sum(msg='m', out='h'))
        h = g.ndata['h']

        h = h + features
        # pair_norm
        norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5)
        norm = norm.to(features.device).unsqueeze(1)
        h = h * norm
        return self.linear(h)


# 数据加载函数
def load_data(dataset):
    if dataset == 'cora':
        data = citegrh.load_cora()
    elif dataset == 'citeseer':
        data = citegrh.load_citeseer()
    else:
        raise ValueError('Unsupported dataset')
    g = data[0]
    u, v = g.edges()
    features = torch.FloatTensor(g.ndata['feat'])
    labels = torch.LongTensor(g.ndata['label'])
    mask = torch.BoolTensor(g.ndata['train_mask'])
    return g, u, v, features, labels, mask


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


# 训练函数
def train(model, g, features, labels, train_mask, val_mask, optimizer, criterion, epochs):
    model.train()
    losses = []  # 用于保存每个epoch的训练损失值
    val_losses = []  # 用于保存每个epoch的验证损失值
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(g, features)
        train_loss = criterion(output[train_mask], labels[train_mask])
        val_loss = criterion(output[val_mask], labels[val_mask])
        train_loss.backward()
        optimizer.step()
        losses.append(train_loss.item())  # 将训练损失值添加到列表中
        val_losses.append(val_loss.item())  # 将验证损失值添加到列表中
        # print('Epoch {:d} | Train Loss {:.4f} | Val Loss {:.4f}'.format(epoch + 1, train_loss.item(), val_loss.item()))
    # 可视化损失与训练次数
    plt.plot(range(epochs), losses, label='Train Loss')
    plt.plot(range(epochs), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


# 验证函数
def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        output = model(g, features)
        pred = output[mask].argmax(dim=1)
        acc = accuracy_score(labels[mask].numpy(), pred.numpy())
        return acc


# 主函数
def main():
    global i
    i = 0
    # 设置随机种子
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = 'cora'  # 可选'cora'或'citeseer'
    g, u, v, features, labels, mask = load_data(dataset)

    model = GCN(features.shape[1], 32, len(np.unique(labels)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 节点分类数据集划分
    n = len(labels)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    n_test = n - n_train - n_val
    indices = np.random.permutation(n)
    train_idx = torch.tensor(indices[:n_train])
    val_idx = torch.tensor(indices[n_train:(n_train + n_val)])
    test_idx = torch.tensor(indices[(n_train + n_val):])
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    train(model, g, features, labels, train_mask, val_mask, optimizer, criterion, epochs=100)


    val_acc = evaluate(model, g, features, labels, val_mask)
    print('Validation Accuracy {:.4f}'.format(val_acc))

    test_acc = evaluate(model, g, features, labels, test_mask)
    print('Test Accuracy {:.4f}'.format(test_acc))
    i=0
    # 链路预测数据划分
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    val_size = int(len(eids) * 0.1)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - val_size - test_size
    train_pos_u, train_pos_v = u[eids[:train_size]], v[eids[:train_size]]
    val_pos_u, val_pos_v = u[eids[train_size:(train_size + val_size)]], v[eids[train_size:(train_size + val_size)]]
    test_pos_u, test_pos_v = u[eids[(train_size + val_size):]], v[eids[(train_size + val_size):]]
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    train_neg_u, train_neg_v = neg_u[neg_eids[:train_size]], neg_v[neg_eids[:train_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[train_size:(train_size + val_size)]], neg_v[
        neg_eids[train_size:(train_size + val_size)]]
    test_neg_u, test_neg_v = neg_u[neg_eids[(train_size + val_size):]], neg_v[neg_eids[(train_size + val_size):]]
    train_g = dgl.remove_edges(g, eids[(train_size + val_size):])
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
    val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=g.number_of_nodes())
    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=g.number_of_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
    pred = DotPredictor()
    del model
    model = GCN(g.ndata['feat'].shape[1], 32, len(np.unique(labels)))
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
    train_losses = []
    val_losses = []
    # train
    all_logits = []
    for e in range(150):
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if e % 5 == 0:
        # print('In epoch {}, loss: {}'.format(e, loss))
        train_losses.append(loss.item())
        val_loss = compute_loss(pred(val_pos_g, h), pred(val_neg_g, h)).item()
        val_losses.append(val_loss)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    with torch.no_grad():
        pos_score = pred(val_pos_g, h)
        neg_score = pred(val_neg_g, h)
        print('Valid AUC:', compute_auc(pos_score, neg_score))
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print('Test AUC:', compute_auc(pos_score, neg_score))


if __name__ == '__main__':
    main()
