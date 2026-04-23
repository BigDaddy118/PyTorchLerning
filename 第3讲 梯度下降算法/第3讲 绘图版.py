import matplotlib.pyplot as plt

# 数据
x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [2.0, 4.0, 6.0, 8.0, 10.0]

w = 1.0

def forward(x):
    return w * x

def cost(xs, ys):
    total = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        total += (y_pred - y) ** 2
    return total / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        grad += 2 * x * (y_pred - y)
    return grad / len(xs)

# 用于记录训练过程中的损失值
epoch_list = []
cost_list = []

print("Predict (before training)", 4, forward(4))

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val

    # 记录数据
    epoch_list.append(epoch)
    cost_list.append(cost_val)

    if epoch % 10 == 0:  # 每10轮打印一次
        print(f"Epoch: {epoch}, w={w:.4f}, cost={cost_val:.6f}")

print("Predict (after training)", 4, forward(4))

# ---------- 绘图部分 ----------
plt.figure(figsize=(12, 4))

# 子图1：损失下降曲线
plt.subplot(1, 2, 1)
plt.plot(epoch_list, cost_list, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Cost (MSE)')
plt.title('Training Loss Curve')
plt.grid(True)

# 子图2：拟合直线与数据点对比
plt.subplot(1, 2, 2)
# 画原始数据点
plt.scatter(x_data, y_data, color='red', label='True Data')
# 画训练前的拟合线 (w=1.0)
w_before = 1.0
y_pred_before = [w_before * x for x in x_data]
plt.plot(x_data, y_pred_before, 'g--', label=f'Before training: w={w_before}')
# 画训练后的拟合线 (当前w值)
y_pred_after = [w * x for x in x_data]
plt.plot(x_data, y_pred_after, 'b-', label=f'After training: w={w:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Fit Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
