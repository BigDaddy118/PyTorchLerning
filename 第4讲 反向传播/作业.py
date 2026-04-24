import torch
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
w1 = torch.tensor([1.0]) # w的初值为1.0
w1.requires_grad = True # 需要计算梯度
w2 = torch.tensor([1.0]) # w的初值为1.0
w2.requires_grad = True # 需要计算梯度
b = torch.tensor([1.0]) # b的初值为1.0
b.requires_grad = True # 需要计算梯度
 
def forward(x):
    return w1*x**2 + w2*x + b  # w是一个Tensor

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l =loss(x,y) # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward() #  backward,compute grad for Tensor whose requires_grad set to True
        print('\tgrad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data   # 权重更新时，注意grad也是一个tensor
        w2.data = w2.data - 0.01 * w2.grad.data   # 权重更新时，注意grad也是一个tensor
        b.data = b.data - 0.01 * b.grad.data   # 权重更新时，注意grad也是一个tensor

        w1.grad.data.zero_() # after update, remember set the grad to zero
        w2.grad.data.zero_() # after update, remember set the grad to zero
        b.grad.data.zero_() # after update, remember set the grad to zero
 
    print('progress:', epoch, l.item()) # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）
