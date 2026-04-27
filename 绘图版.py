import torch
import matplotlib.pyplot as plt

# ------------------ 数据集 ------------------
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

# ------------------ 模型 ------------------
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionModel()

# ------------------ 损失函数与优化器 ------------------
criterion = torch.nn.BCELoss(size_average=False)  # 注意新版PyTorch需改为 reduction='sum'
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ------------------ 绘图准备 ------------------
plt.ion()                     # 打开交互模式
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# 左图：损失曲线
loss_list = []
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
line_loss, = ax1.plot([], [], 'b-')

# 右图：数据与拟合曲线
ax2.scatter(x_data.numpy(), y_data.numpy(), color='red', label='True data')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Logistic Regression Fit')
x_line = torch.linspace(0, 4, 100).view(-1, 1)   # 用于绘制光滑曲线
line_fit, = ax2.plot([], [], 'g-', label='Fitted sigmoid')
ax2.legend()

# ------------------ 训练 ------------------
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录损失
    loss_list.append(loss.item())

    # 每 50 轮更新一次图像（避免更新太频繁导致卡顿）
    if epoch % 50 == 0 or epoch == 999:
        # 更新损失曲线
        line_loss.set_data(range(len(loss_list)), loss_list)
        ax1.relim()
        ax1.autoscale_view()

        # 更新拟合曲线
        with torch.no_grad():
            y_line = model(x_line).numpy()
        line_fit.set_data(x_line.numpy(), y_line)

        plt.pause(0.01)   # 短暂暂停以刷新图像

plt.ioff()                # 关闭交互模式
plt.show()                # 训练结束后保留图像