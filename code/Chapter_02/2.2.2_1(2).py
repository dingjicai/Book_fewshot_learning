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
    data = np.load('mnist.npz')
    train_x = data['x_train']
    train_y = data['y_train']
    test_x = data['x_test']
    test_y = data['y_test']
    
    #训练数据
    ds_train = tf.data.Dataset.from_tensor_slices((train_x, train_x))
    ds_train = ds_train.map(preprocess_train)
    ds_train = ds_train.shuffle(buffer_size).batch(batch_size)
    #测试数据
    ds_test = tf.data.Dataset.from_tensor_slices((test_x, test_x))
    ds_test = ds_test.map(preprocess_train)
    ds_test = ds_test.batch(batch_size)
    #推理数据
    ds_predict = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds_predict = ds_predict.map(preprocess_predict)
    ds_predict = ds_predict.batch(1)
    return ds_train, ds_test, ds_predict
#数据预处理
def preprocess_train(x, y):
    #数据形状由28*28变成28*28*1
    x = x[:, :, np.newaxis]
    # 归一化
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.float32) / 255
    
    return x, y

def preprocess_predict(x, y):
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
    dense1 = layers.Dense(120, activation='relu')(flatten)
    dense2 = layers.Dense(2)(dense1)
    dense3 = layers.Dense(120, activation='relu')(dense2)
    reshape = layers.Reshape((1, 1, 120))(dense3)
    T_conv1 = layers.Conv2DTranspose(filters=16, kernel_size=5, activation='relu')(reshape)
    upsampling1 = layers.UpSampling2D((2,2 ))(T_conv1)
    T_conv2 = layers.Conv2DTranspose(filters=6, kernel_size=5, activation='relu')(upsampling1)
    upsampling2 = layers.UpSampling2D((2,2 ))(T_conv2)
    T_conv3 = layers.Conv2DTranspose(filters=1, kernel_size=5, padding='same')(upsampling2)
    #输出层
    outputs = T_conv3
    #构建模型
    model = Model(inputs, outputs)
    return model
#设置超参
epochs = 5
learn_rate = 0.001
batch_size = 32
buffer_size = 1000
#获取训练 测试数据 
ds_train, ds_test, ds_predict = get_data(buffer_size, batch_size)
#搭建并初始化网络
model = LeNet()
noise = tf.random.normal([1, 28, 28, 1])
noise_out = model(noise)
print(noise_out.shape)
model.summary()
#定义优化器
optimizer = optimizers.Adam(learning_rate=learn_rate)
#定义目标函数
loss = tf.keras.losses.MSE
#编译模型
model.compile(optimizer=optimizer, loss=loss)
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
#将图片shape从28*28*1变为1*28*28*1，增加一个batch维度，以匹配模型输入格式要求
img_batch = np.expand_dims(img.astype('float32'), axis=0)
#执行推理，此处predict_on_batch返回的是一个list，取出其中数据获得预测结果
out = model_loaded.predict_on_batch(img_batch)[0]

#可视化图片
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(20,5))
for i in range(7):
    plt.subplot(2, 7, i+1)
    img = list(ds_test.take(7).as_numpy_iterator())[i][0][0, :, :, :]
    plt.imshow(np.squeeze(img))
for i in range(7):
    plt.subplot(2, 7, i+8)
    img = list(ds_test.take(7).as_numpy_iterator())[i][0][0, :, :, :]
    img_batch = np.expand_dims(img.astype('float32'), axis=0)
    out = model_loaded.predict_on_batch(img_batch)[0]
    plt.imshow(np.squeeze(out))
plt.show()

#提取推理数据前300张图片的编码向量（长度2）
from tensorflow.keras import backend as K

layer_sub = K.function([model.layers[0].input], [model.get_layer(name='dense_1').output])
vec_en = np.zeros((300, 2), dtype=float)
label = np.zeros(300, dtype=float)
img_300 = np.zeros((300, 28*28), dtype=float)
for i, (img, y) in enumerate(ds_predict.take(300)):
    img_300[i, :] = img.numpy().flatten()
    vec_en[i, :] = layer_sub([img])[0][0]
    label[i] = y.numpy().argmax()
#可视化编码效果
fig = plt.figure(figsize=(20, 10))
colors = ["k", 'gray', 'r', 'peru', 'c', 'green', 'cyan',
          'm', 'b', 'royalblue']
makers = ["o", 's', 'p', '*', '+', 'x', 'D',
          '2', 'H', '.']
for i in range(10):
    one_num = vec_en[label == i, :]
    plt.scatter(one_num[:, 0], one_num[:, 1], c=colors[i], s=400, marker=makers[i])
plt.show()


fig1, ax = plt.subplots() 
for i in range(300):
    ax.text(vec_en[i,0],vec_en[i,1],str(int(label[i])),color=plt.cm.Set1(label[i]/10.),
            fontdict={'weight': 'bold', 'size': 12})
ax.set(xlim =(min(vec_en[:,0])-0.1, max(vec_en[:,0])+0.1), ylim =(min(vec_en[:,1])-0.1, max(vec_en[:,1])+0.1)) 
plt.show()


from sklearn.decomposition import PCA
fig = plt.figure(figsize=(20, 10))
pca = PCA(n_components=2)
pca = pca.fit(img_300)
img_300_pca = pca.transform(img_300)
colors = ["k", 'gray', 'r', 'peru', 'c', 'green', 'cyan',
          'm', 'b', 'royalblue']
makers = ["o", 's', 'p', '*', '+', 'x', 'D',
          '2', 'H', '.']
for i in range(10):
    one_num = img_300_pca[label == i, :]
    plt.scatter(one_num[:, 0], one_num[:, 1], c=colors[i], s=400, marker=makers[i])
plt.show()


fig2, ax = plt.subplots() 
for i in range(300):
    ax.text(img_300_pca[i,0],img_300_pca[i,1],str(int(label[i])),color=plt.cm.Set1(label[i]/10.),
            fontdict={'weight': 'bold', 'size': 12})
ax.set(xlim =(-5, 6.5), ylim =(-4.5, 6)) 
plt.show()
