import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 1. 定义数据集类
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

# 2. 定义模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x  

if __name__ == '__main__':    
    
    dataset = DiabetesDataset('diabetes.csv.gz')
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)    

    model = Model()

    # 3. 定义损失函数和优化器
    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 4. 训练模型
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            #1. 获取输入数据
            inputs, labels = data
            #2. 前向传播
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            #3. 后向传播和优化
            optimizer.zero_grad()
            loss.backward()
            #4. 更新参数
            optimizer.step()
