import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt                      # 新增：用于绘图

# ------------------ 数据集 ------------------
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='../dataset/mnist/', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(
    root='../dataset/mnist/', train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# ------------------ Inception 模块 ------------------
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)            # 输出通道：16+24+24+24=88


# ------------------ 主网络 ------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)   # 28→24
        self.incep1 = InceptionA(in_channels=10)       # 24→24，输出88通道
        self.mp = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)  # 12→8
        self.incep2 = InceptionA(in_channels=20)       # 8→8，输出88通道

        self.fc = nn.Linear(1408, 10)                  # 88*4*4 = 1408

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))   # 10×12×12
        x = self.incep1(x)                  # 88×12×12
        x = F.relu(self.mp(self.conv2(x)))  # 20×8×8 → mp → 20×4×4
        x = self.incep2(x)                  # 88×4×4
        x = x.view(in_size, -1)             # 展平
        x = self.fc(x)
        return x

model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# ------------------ 训练（记录 loss） ------------------
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:      # 每100个batch打印一次
            avg_loss = running_loss / 100
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data),
                len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader),
                avg_loss))
            train_losses.append(avg_loss)   # 记录平均损失
            running_loss = 0.0

# ------------------ 测试（记录准确率） ------------------
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = 100. * correct / total
    test_accuracies.append(acc)              # 记录准确率
    print('Accuracy on test set: {:.2f} %\n'.format(acc))

# ------------------ 主程序：训练 + 绘图 ------------------
if __name__ == '__main__':
    train_losses = []      # 存储每100个batch的平均损失
    test_accuracies = []   # 存储每个epoch的测试准确率

    for epoch in range(1, 11):
        train(epoch)
        test()

    # 绘制损失曲线（横轴为 batch 组次）
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', linewidth=0.5)
    plt.xlabel('Batch group (per 100 batches)')
    plt.ylabel('Average Loss')
    plt.title('Training Loss')

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='s', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy')
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.show()
