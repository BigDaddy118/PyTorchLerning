import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import gzip

# ========== 1. 定义数据集类 ==========
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        # 使用 gzip 直接读取 .gz 文件（如果文件已解压，改为普通 open）
        with gzip.open(filepath, 'rt') as f:
            xy = np.loadtxt(f, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# ========== 2. 定义模型 ==========
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)          # 等同于之前的 self.sigmoid(x)
        return x

# ========== 主程序 ==========
if __name__ == '__main__':
    # 数据集与加载器
    dataset = DiabetesDataset('diabetes.csv.gz')
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

    model = Model()

    # 损失函数和优化器（修正了弃用参数）
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 用于记录每个 epoch 的平均损失
    epoch_losses = []

    # ========== 3. 训练模型 ==========
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0.0   # 累加本 epoch 中所有 batch 的损失
        batch_count = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # 前向传播
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # （可选）打印每个 batch 的详细信息，调试时再用
            # print(f'Epoch {epoch}, Batch {i}, Loss {loss.item():.4f}')

        # 计算并保存平均损失
        avg_loss = total_loss / batch_count
        epoch_losses.append(avg_loss)
        print(f'Epoch {epoch:3d}  |  Average Loss: {avg_loss:.6f}')

    # ========== 4. 绘制损失曲线 ==========
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), epoch_losses, 'b-', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print('训练完成，图形已显示。')
