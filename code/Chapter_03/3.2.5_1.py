#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:23:33 2022

@author: dingjc
"""


import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

#加载数据
data = np.load('mnist.npz')
train_x = data['x_train']
train_x = train_x[:, np.newaxis, :, :]/255  
train_x = torch.from_numpy(train_x).float()
train_x = F.pad(train_x, (2,2,2,2), mode='constant', value=0.0)

#设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 1
)
diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)
diffusion = diffusion.to(device)

#超参数
iter_number =10000
batch = 32
learn_rate = 1e-4
adam_betas = (0.9, 0.99)
#优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate,  betas = adam_betas)

#训练
for i in tqdm(range(iter_number)):
    index = np.random.randint(0, train_x.shape[0], batch)
    training_images = train_x[index]
    training_images = training_images.to(device)
    optimizer.zero_grad()
    loss = diffusion(training_images)
    loss.backward()
    optimizer.step()

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 1, 32, 32)
fig = plt.figure(figsize=(10,4))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(np.squeeze(sampled_images.cpu()[i,]))
plt.show()