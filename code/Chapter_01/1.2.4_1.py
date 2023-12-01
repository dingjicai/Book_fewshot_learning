# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 10:44:05 2022
@author: DingJiCai
"""
import tensorflow as tf
from tensorflow.keras import Input, optimizers, layers, Model
import numpy as np
#加载手写数字数据
def get_data(buffer_size,batch_size):
    #读取数据
    data = np.load('mnist.npz')
    train_x = data['x_train']
    train_y = data['y_train']
    test_x = data['x_test']
    test_y = data['y_test']
    #生成训练数据
    ds_train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    ds_train = ds_train.map(preprocess)
    ds_train = ds_train.shuffle(buffer_size).batch(batch_size)
    #生成测试数据
    ds_test = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds_test = ds_test.map(preprocess)
    ds_test = ds_test.batch(batch_size)
    return ds_train, ds_test
#数据预处理
def preprocess(x, y):
    #数据形状由28*28变成28*28*1，增加信号通道维度
    x = x[:, :, np.newaxis]
    #数据归一化，标签独热编码
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y
#搭建网络
def LeNet():
    #输入层
    inputs = Input(shape=(28, 28, 1))
    #隐层
    conv1 = layers.Conv2D(filters=6, kernel_size=5, padding='same', activation='relu')(inputs)
    pool1 = layers.AveragePooling2D((2, 2))(conv1)
    conv2 = layers.Conv2D(filters=16, kernel_size=5, activation='relu')(pool1)
    pool2 = layers.AveragePooling2D((2, 2))(conv2)
    conv3 = layers.Conv2D(filters=120, kernel_size=5, activation='relu')(pool2)
    flatten = layers.Flatten()(conv3)
    dense1 = layers.Dense(84, activation='relu')(flatten)
    #输出层
    outputs = layers.Dense(10, activation='softmax')(dense1)
    #构建模型
    model = Model(inputs, outputs)
    return model
#设置超参
epochs = 5
learn_rate = 0.001
batch_size = 32
buffer_size = 1000
#获取训练和测试数据
ds_train, ds_test = get_data(buffer_size, batch_size)
#获取并初始化网络
model = LeNet()
noise = tf.random.normal([1, 28, 28, 1])
noise_out = model(noise)
print(noise_out)
#网络结构展示
model.summary()
#定义优化器
optimizer = optimizers.Adam(learning_rate=learn_rate)
#定义目标函数
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
#编译模型
model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
#训练模型
history = model.fit(x=ds_train, validation_data=ds_test, epochs=epochs)
#模型评估
model.evaluate(ds_test, verbose=1)
#保存模型
model.save('./output/mnist_tensorflow')
#加载模型
model_loaded = tf.keras.models.load_model('output/mnist_tensorflow')
#从测试集中取出一张图片
img = list(ds_test.take(1).as_numpy_iterator())[0][0][0, :, :, :]
label = list(ds_test.take(1).as_numpy_iterator())[0][1][0, :]
#将图片shape从28*28*1变为1*28*28*1，增加batch维度，以匹配模型输入格式要求
img_batch = np.expand_dims(img.astype('float32'), axis=0)
#推理并打印结果，此处predict_on_batch返回的是一个list，取出其中数据获得预测结果
out = model_loaded.predict_on_batch(img_batch)[0]
pred_label = out.argmax()
print('true label: {}, predict label: {}'.format(label.argmax(), pred_label))
#可视化图片
from matplotlib import pyplot as plt
plt.imshow(img)
plt.show()