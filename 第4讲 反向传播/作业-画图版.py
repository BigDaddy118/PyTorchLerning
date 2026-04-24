import torch
import matplotlib.pyplot as plt

# 数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 参数初始化
w1 = torch.tensor([1.0], requires_grad=True)
w2 = torch.tensor([1.0], requires_grad=True)
b  = torch.tensor([1.0], requires_grad=True)

def forward(x):
    return w1 * x**2 + w2 * x + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# 记录每个epoch的平均损失
epoch_losses = []

print("predict (before training)", 4, forward(4).item())

# 训练
for epoch in range(100):
    total_loss = 0.0
    for x, y in zip(x_data, y_data):
        l = loss(x, y)           # 前向计算损失
        l.backward()             # 反向传播求梯度
        
        # 梯度更新
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data  = b.data  - 0.01 * b.grad.data
        
        # 梯度清零
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
        
        total_loss += l.item()
    
    avg_loss = total_loss / len(x_data)
    epoch_losses.append(avg_loss)
    
    # 保留原格式，但显示当前epoch的平均损失
    print(f'progress: {epoch}  avg_loss: {avg_loss:.6f}', end='')
    print(f'  |  w1={w1.item():.4f}, w2={w2.item():.4f}, b={b.item():.4f}')

print("\nFinal parameters: w1 =", w1.item(), "w2 =", w2.item(), "b =", b.item())
print("predict (after training) x=4 :", forward(4).item())

# ---------- 绘图 ----------
plt.figure(figsize=(12, 4))

# 左图：损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(len(epoch_losses)), epoch_losses, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss Curve (Quadratic Model)')
plt.grid(True)

# 右图：数据散点与拟合曲线
plt.subplot(1, 2, 2)
plt.scatter(x_data, y_data, color='red', label='Training data')
# 生成连续x用于画曲线
x_curve = torch.linspace(0, 5, 200)
y_curve = forward(x_curve).detach().numpy()
plt.plot(x_curve.numpy(), y_curve, 'g-', label=f'Fitted: y = {w1.item():.3f}$x^2$ + {w2.item():.3f}$x$ + {b.item():.3f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Fit (should approach y=2x)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
