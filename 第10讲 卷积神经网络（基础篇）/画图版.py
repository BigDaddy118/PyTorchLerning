import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt   # 用于绘图

# 准备数据集
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='../dataset/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='../dataset/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 50)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

model = Net()

# loss函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 训练模型（修改版：返回当前epoch的平均损失）
def train(epoch):
    running_loss = 0.0
    total_batches = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
    
    # 返回整个epoch的平均损失
    return running_loss / total_batches

# 测试函数（修改版：返回准确率）
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %.2f %%' % acc)
    return acc

if __name__ == '__main__':
    train_losses = []
    test_accuracies = []
    epochs = 10

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        avg_loss = train(epoch)
        acc = test()
        train_losses.append(avg_loss)
        test_accuracies.append(acc)

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, marker='o', color='blue', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), test_accuracies, marker='s', color='green', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Curve')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
