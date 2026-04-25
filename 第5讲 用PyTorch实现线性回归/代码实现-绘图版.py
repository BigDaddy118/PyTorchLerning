import torch
import matplotlib.pyplot as plt

# 准备数据集
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# 设计模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 输入1维，输出1维

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

# 损失函数和优化器
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ---------- 绘图设置 ----------
plt.ion()                    # 打开交互模式
fig, ax = plt.subplots()    # 创建图窗
ax.set_xlim(0, 5)
ax.set_ylim(0, 8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Linear Regression Fitting')

# 画原始数据点（散点）
ax.scatter(x_data.numpy(), y_data.numpy(), color='red', label='True data')

# 用于动态更新直线的对象
line, = ax.plot([], [], 'b-', label='Fitted line')
ax.legend()

# ---------- 训练循环 ----------
for epoch in range(100):
    y_pred = model(x_data)                     # 前向预测
    loss = criterion(y_pred, y_data)           # 计算损失

    optimizer.zero_grad()                      # 梯度清零
    loss.backward()                            # 反向传播
    optimizer.step()                           # 参数更新

    # 每 5 个 epoch 更新一次图
    if epoch % 5 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        # 用当前模型计算一条直线用于绘图（取 0~5 之间若干点）
        x_line = torch.linspace(0, 5, 100).view(-1, 1)
        y_line = model(x_line).detach().numpy()

        # 更新直线数据
        line.set_xdata(x_line.numpy())
        line.set_ydata(y_line)

        plt.pause(0.01)   # 短暂暂停以刷新图形

# 训练结束后关闭交互模式，并显示最终图形
plt.ioff()
plt.show()

# 打印最终参数
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

# 测试
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data.item())
