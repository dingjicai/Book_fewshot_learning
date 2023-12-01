# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 10:44:05 2022
@author: DingJiCai
"""
import tensorflow as tf
from tensorflow.keras import Input, optimizers, layers, Model
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
#加载手写数字数据
def get_data(buffer_size,batch_size):
    data = np.load('mnist.npz')
    train_x = data['x_train']
    train_y = data['y_train']
    test_x = data['x_test']
    test_y = data['y_test']

    train_x = np.expand_dims(train_x, axis=-1).astype(np.float32)/255.0
    test_x = np.expand_dims(test_x, axis=-1).astype(np.float32) / 255.0

    mlb = MultiLabelBinarizer()
    task_a_y = mlb.fit_transform(np.expand_dims(train_y, -1))
    task_b_y = np.zeros((train_y.shape[0], 2), dtype='uint8')
    y_6 = (train_y>=7)
    y_odd = (train_y % 2 == 1)
    task_b_y[y_6, 0] = 1
    task_b_y[y_odd, 1] = 1

    test_task_a_y = mlb.fit_transform(np.expand_dims(test_y, -1))
    test_task_b_y = np.zeros((test_y.shape[0], 2), dtype='uint8')
    y_6 = (test_y>=7)
    y_odd = (test_y % 2 == 1)
    test_task_b_y[y_6, 0] = 1
    test_task_b_y[y_odd, 1] = 1

    #训练数据
    ds_train = tf.data.Dataset.from_tensor_slices((train_x, (task_a_y, task_b_y)))
    ds_train = ds_train.shuffle(buffer_size).batch(batch_size)
    #测试数据
    ds_test = tf.data.Dataset.from_tensor_slices((test_x, (test_task_a_y, test_task_b_y)))
    ds_test = ds_test.batch(batch_size)
    return ds_train, ds_test

#数据预处理
def preprocess(x, y):
    #数据形状由32*32变成32*32*1
    x = x[:, :, np.newaxis]
    #归一化
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y
#搭建网络
def LeNet():
    #输入层
    inputs = Input(shape=(28, 28, 1))
    conv1 = layers.Conv2D(filters=6, kernel_size=5, padding='same', activation='relu')(inputs)
    pool1 = layers.AveragePooling2D((2, 2))(conv1)
    conv2 = layers.Conv2D(filters=16, kernel_size=5, activation='relu')(pool1)
    pool2 = layers.AveragePooling2D((2, 2))(conv2)
    conv3 = layers.Conv2D(filters=120, kernel_size=5, activation='relu')(pool2)
    flatten = layers.Flatten()(conv3)
    dense1 = layers.Dense(84, activation='relu')(flatten)
    #输出层
    outputs_a = layers.Dense(10, activation='softmax')(dense1)
    outputs_b = layers.Dense(2, activation='sigmoid')(dense1)
    #构建模型
    model = Model(inputs, (outputs_a, outputs_b))
    return model
#设置超参
epochs = 1
learn_rate = 0.001
batch_size = 32
buffer_size = 1000
#获取训练和测试数据
ds_train, ds_test = get_data(buffer_size, batch_size)
#搭建并初始化网络
model = LeNet()
noise = tf.random.normal([1, 28, 28, 1])
noise_out = model(noise)
print(noise_out)
model.summary()
#定义优化器
optimizer = optimizers.Adam(learning_rate=learn_rate)
#定义目标函数
def loss_fn(preds, labels):
    loss_a = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true=labels[0], y_pred=preds[0])
    loss_b = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true=labels[1], y_pred=preds[1])
    return loss_a + loss_b
#训练模型
def fit():
    for epoch_idx in range(epochs):
        for batch_idx, (data, labels) in enumerate(ds_train):
            train_loss = train_step(data, labels)
            if batch_idx % 100 == 0:
                val_loss = 0
                for index, (val_data, val_labels) in enumerate(ds_test):
                    val_loss += loss_fn(model(val_data, training=False), val_labels)
                print('epoch=', epoch_idx+1, 'train loss=', train_loss.numpy(), 'val loss=', val_loss.numpy()/(index+1))
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        preds = model(inputs, training=True)
        loss = loss_fn(preds, labels)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
fit()

#从测试集中取出一个batch
img = ds_test.take(1)
one_batch = list(img.as_numpy_iterator())[0]
#执行推理并打印结果[10个样本]
out = model.predict_on_batch(one_batch[0])
#0.5为阈值判别
out[0][out[0] > 0.5] = 1
out[0][out[0] <= 0.5] = 0
out[1][out[1] > 0.5] = 1
out[1][out[1] <= 0.5] = 0
print(out[0][0:10])#任务a输出
print(out[1][0:10])#任务b输出
print(one_batch[1][0][0:10])#任务a标签
print(one_batch[1][1][0:10])#任务b标签
