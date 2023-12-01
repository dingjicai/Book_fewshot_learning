#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:31:36 2022

@author: DingJicai
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import numpy as np

#超参数
image_size = 28*28*1
h_dim = 400
z_dim = 20
epochs = 30
batch_size = 32
learning_rate = 0.001
#获取数据
class get_data(Dataset):
    def __init__(self, train_data=True):
        self.train_data = train_data
        self.data = np.load('mnist.npz')
        self.train_x = torch.tensor(self.data['x_train'].astype(np.float32))
 
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
        if not self.train_data:
            x = self.test_x[idx, :, :]
            x = x[np.newaxis, :, :]
        return x/255
    
train_loader = DataLoader(dataset=get_data(), batch_size=batch_size, shuffle=True)

#构建VAE模型
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
 
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
 
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.rand_like(std)
        return mu + eps * std
 
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))
 
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
#loss函数 
def loss_function(x_reconst, x, mu, log_var): 
    BCE_loss = nn.BCELoss(reduction='sum')
    reconstruction_loss = BCE_loss(x_reconst, x)
    KL_divergence = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mu ** 2)
    return reconstruction_loss + KL_divergence

image = iter(train_loader).next()
fig = plt.figure()
for i in range(12):
    plt.subplot(2, 6, i + 1)
    plt.imshow(image[i][0], cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
plt.show()
#设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#模型
model = VAE().to(device)
#优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#开始训练
losses = []
for epoch in range(30) :
    train_loss = 0
    train_acc = 0
    model.train()
    for index, imgs in enumerate(train_loader) :
        imgs = imgs.to(device)
        real_imgs = torch.flatten(imgs, start_dim=1)
        #前向传播
        gen_imgs, mu, log_var = model(real_imgs)
        loss = loss_function(gen_imgs, real_imgs, mu, log_var)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #记录误差
        train_loss += loss.item()
    print('epoch: {}, loss: {}'.format(epoch+1, train_loss / len(train_loader)))
    losses.append(train_loss / len(train_loader))
    fake_images = gen_imgs.view(-1, 1, 28, 28)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 6, i + 1)
        plt.imshow(image.cpu().detach().numpy()[i][0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    for i in range(6):
        plt.subplot(2, 6, i + 7)
        plt.imshow(fake_images.cpu().detach().numpy()[i][0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()

plt.title('trainloss')
plt.plot(np.arange(len(losses)), losses, linewidth=1.5, linestyle='dashed', label='train_losses')
plt.xlabel('epoch')
plt.legend()
plt.show()