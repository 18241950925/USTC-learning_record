import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义模型
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)  # 批量归一化层
        self.norm3 = nn.BatchNorm2d(128)  # 批量归一化层

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.norm1(x)
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.norm3(x)  # 应用批量归一化
        x = self.maxpool(x)
        #print(x.shape)            #根据这个值修改view参数以及fc1的输入参数
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 数据加载与预处理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 64
data_root_dir = './../data'
dataset = torchvision.datasets.CIFAR10(root=data_root_dir, train=True, download=False, transform=transform)
# 划分训练集与验证集
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
# 初始化模型、损失函数和优化器
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)   # 课上讲的效果最好的优化器


# 训练模型
def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    return train_loss


# 验证模型
def validate(model, valid_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    valid_loss = running_loss / len(valid_loader.dataset)
    accuracy = correct / total
    return valid_loss, accuracy


# 训练模型
num_epochs = 30
learning_rate = 0.01
learning_rate_decay = 0.1
decay_epoch = 15
train_losses = []
valid_losses = []
valid_accuracies = []
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_epoch, gamma=learning_rate_decay)
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    valid_loss, accuracy = validate(model, valid_loader, criterion)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_accuracies.append(accuracy)
    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {100 * accuracy:.2f}%')
    scheduler.step()

plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# 在测试集上测试
test_dataset = torchvision.datasets.CIFAR10(root=data_root_dir, train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_loss, test_accuracy = validate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {100 * test_accuracy:.2f}%')
