# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 10:44:05 2022
@author: DingJiCai
"""
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchsummary import summary
import torchmetrics
#获取数据
class get_data(Dataset):
    def __init__(self, train_data=True):
        self.train_data = train_data
        self.data = np.load('mnist.npz')
        self.train_x = torch.tensor(self.data['x_train'].astype(np.float32))
        self.train_y = torch.tensor(self.data['y_train'].astype(np.int64))
        self.test_x = torch.tensor(self.data['x_test'].astype(np.float32))
        self.test_y = torch.tensor(self.data['y_test'].astype(np.int64))
    def __len__(self):
        if self.train_data:
            length = self.train_x.shape[0]
        if not self.train_data:
            length = self.test_x.shape[0]
        return length
    def __getitem__(self, idx):
        if self.train_data:
            x = self.train_x[idx, :, :]
            x = x[np.newaxis, :, :]
            y = self.train_y[idx]
            y = F.one_hot(y, num_classes=10)
        if not self.train_data:
            x = self.test_x[idx, :, :]
            x = x[np.newaxis, :, :]
            y = self.test_y[idx]
            y = F.one_hot(y, num_classes=10)
        return x/255, y.float()
#搭建网络LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 120, 5),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.Softmax()
        )
    def forward(self, x):
        feature = self.conv(x)
        output = self.fc(feature.view(x.shape[0], -1))
        return output
#定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#设置超参
epochs = 5
learn_rate = 0.001
batch_size = 32
#获取训练和测试数据
ds_train = get_data(True)
ds_test = get_data(False)
train_loader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=False)
#搭建并初始化网络
model = LeNet()
model = model.to(device)
noise = torch.randn([1, 1, 28, 28]).to(device)
noise_out = model(noise)
print(noise_out)
print(summary(model, input_size=(1, 28, 28)))
#定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
#定义目标函数
loss_func = nn.CrossEntropyLoss()
#训练
for epoch in range(1, epochs+1):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        #梯度清零
        optimizer.zero_grad()
        #损失值
        outputs = model(data)
        loss = loss_func(outputs, label)
        #反向传播求梯度
        loss.backward()
        optimizer.step()
        #记录误差
        if batch_idx % 500 == 0:
            print('epoch {},Train loss {:.6f},Dealed/Records:{}/{}'.format(epoch, loss/batch_size, (batch_idx+1)*batch_size,60000))
    model.eval()
    test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    for val_step, (data, label) in enumerate(test_loader):
        data = data.to(device)
        label = label.to(device)
        with torch.no_grad():
            outputs = model(data)
            test_acc(outputs.argmax(1), label.argmax(1))
    total_acc = test_acc.compute()
    print("val_acc:", total_acc.item())
    test_acc.reset()
#保存模型
torch.save(model, './output/mnist_torch.pt')
#加载模型
model_loaded = torch.load('output/mnist_torch.pt')
model_loaded = model_loaded.to(device)
#从测试集中取出一张图片
img, label = ds_test[0]
#将图片shape从1*28*28变为1*1*28*28，增加batch维度，以匹配模型输入格式要求
img_batch = torch.unsqueeze(img, dim=0).to(device)
#执行推理并打印结果，此处out是一个list，取出其中数据获得预测结果
out = model_loaded(img_batch)[0]
pred_label = out.argmax()
print('true label: {}, predict label: {}'.format(label.argmax(), pred_label))
#可视化图片
from matplotlib import pyplot as plt
plt.imshow(np.squeeze(img.numpy()))
plt.show()