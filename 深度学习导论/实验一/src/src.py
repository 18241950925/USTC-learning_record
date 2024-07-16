import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CSVDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.x = torch.FloatTensor(data['x'].values).unsqueeze(-1).to(device)
        self.y = torch.FloatTensor(data['y'].values).to(device)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        # self.fc4 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        # x = self.relu(self.fc4(x))
        x = self.fc4(x)
        x = x.squeeze(-1)
        return x


model = FeedForwardNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)


def train(model, train_loader, optimizer, criterion, epochs=300):
    model.train()
    epoch_losses = []  # 用于收集每个epoch的损失
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
    return epoch_losses  # 返回每个epoch的损失列表


N_values = [2000]

for N in N_values:
    # x, y = generate_dataset(N)
    # train_set, valid_set, test_set = split_dataset(x, y)
    # 创建数据集
    train_set = CSVDataset(f'data/train_{N}.csv')
    valid_set = CSVDataset(f'data/valid_{N}.csv')
    test_set = CSVDataset(f'data/test_{N}.csv')
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    epoch_losses = train(model, train_loader, optimizer, criterion, epochs=300)

    # 绘制损失图
    """plt.plot(epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()"""

    model.eval()
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
    print(f'Validation MSE for N={N}: {val_loss.item()}')

    # Plot the approximation
    x = valid_set.x
    y = valid_set.y
    plt.figure()
    plt.scatter(x.cpu().numpy(), y.cpu().numpy(), label='Original')
    model.eval()
    with torch.no_grad():
        plt.scatter(x.cpu().numpy(), model(x).cpu().numpy(), label='Predicted')
    plt.title(f'Approximation for N={N}')
    plt.legend()
    plt.show()
    """# Test on test set
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, targets)
    test_loss /= len(test_loader)
    print(f'Test MSE for N={N}: {test_loss.item()}')"""
