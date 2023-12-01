# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 10:44:05 2022
@author: DingJiCai
"""
import paddle
from paddle import nn
import numpy as np
from paddle.io import Dataset,DataLoader
import paddle.nn.functional as F
#获取数据
class get_data(Dataset):
    def __init__(self, train_data=True):
        self.train_data = train_data
        self.data = np.load('mnist.npz')
        self.train_x = paddle.to_tensor(self.data['x_train'].astype(np.float32))
        self.train_y = paddle.to_tensor(self.data['y_train'].astype(np.int64))
        self.test_x = paddle.to_tensor(self.data['x_test'].astype(np.float32))
        self.test_y = paddle.to_tensor(self.data['y_test'].astype(np.int64))
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
            y = paddle.squeeze(F.one_hot(y, num_classes=10))
        if not self.train_data:
            x = self.test_x[idx, :, :]
            x = x[np.newaxis, :, :]
            y = self.test_y[idx]
            y = paddle.squeeze(F.one_hot(y, num_classes=10))
        return x/255, y
#搭建网络
class LeNet(nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        #构建 features 子网，用于对输入图像进行特征提取
        self.features = nn.Sequential(
            nn.Conv2D(1, 6, 5, stride=1, padding="SAME"),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(6, 16, 5, stride=1, padding="VALID"),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(16, 120, 5, stride=1, padding="VALID"),
            nn.ReLU()
            )
        #构建 linear 子网，用于分类
        self.linear = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
            nn.Softmax()
            )
    #执行前向计算
    def forward(self, inputs):
        x = self.features(inputs)
        x = paddle.flatten(x, 1)
        x = self.linear(x)
        return x
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
model_LeNet = LeNet()
noise = paddle.rand([1, 1, 28, 28])
noise_out = model_LeNet(noise)
print(noise_out)
print(paddle.summary(model_LeNet, (1, 1, 28, 28)))
model = paddle.Model(model_LeNet)
#模型训练的配置准备，准备损失函数，优化器和评价指标
model.prepare(paddle.optimizer.Adam(learning_rate=learn_rate, parameters=model.parameters()),
              paddle.nn.CrossEntropyLoss(soft_label=True),
              paddle.metric.Accuracy())
#模型训练
model.fit(train_loader, epochs=epochs, verbose=1)
#模型评估
model.evaluate(test_loader, verbose=1)
#保存模型
model.save('./output/mnist_paddle')
#加载模型
model_loaded = paddle.Model(model_LeNet)
model_loaded.load('output/mnist_paddle')
#从测试集中取出一张图片
img, label = ds_test[0]
#将图片shape从1*28*28变为1*1*28*28，增加一个batch维度，以匹配模型输入格式要求
img_batch = np.expand_dims(img.astype('float32'), axis=0)
#执行推理并打印结果，此处predict_batch返回的是一个list，取出其中数据获得预测结果
out = model_loaded.predict_batch(img_batch)[0]
pred_label = out.argmax()
print('true label: {}, predict label: {}'.format(label.argmax(), pred_label))
#可视化图片
from matplotlib import pyplot as plt
plt.imshow(img[0])
plt.show()
