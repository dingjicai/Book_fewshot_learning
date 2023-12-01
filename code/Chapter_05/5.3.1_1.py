# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 10:44:05 2022
@author: DingJiCai
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np

#设置超参
num_labeled = 300
epochs = 2000
learn_rate = 0.0001
batch_size = 32
img_shape = (28, 28, 1)
noise_dim = 100
num_classes = 10
verbose_interval = 40

#加载手写数字数据
def get_data(num_labeled):
    data = np.load('mnist.npz')
    x_ = data['x_train']
    y_ = data['y_train']
    test_x_ = data['x_test']
    test_y_ = data['y_test']
    
    x_labeled = x_[0:num_labeled, ]
    y_labeled = y_[0:num_labeled, ]
    x_unlabeled = x_[num_labeled:, ]
    y_unlabeled = y_[num_labeled:, ]
    
    x_labeled = preprocess_imgs(x_labeled)
    y_labeled = preprocess_labels(y_labeled)
    x_unlabeled = preprocess_imgs(x_unlabeled)
    y_unlabeled = preprocess_labels(y_unlabeled)
     
    x_test = preprocess_imgs(test_x_)
    y_test= preprocess_labels(test_y_)
    
    return x_labeled, y_labeled, x_unlabeled, y_unlabeled, x_test, y_test

#数据预处理
def preprocess_imgs(x):
    #数据形状由28*28变成28*28*1
    x = x[:, :, :, np.newaxis]
    #数据规则化[-1 ,1]
    x.astype(np.float)
    x = (x - 127.5) / 127.5
    return x

def preprocess_labels(y):
    #标签独热编码
    y.astype(np.int32)
    y = np.eye(num_classes)[y]
    return y

#生成器
def generator_net(noise_dim):
    model = Sequential()
    #通过一个全连接层改变输入为一个7*7*256的张量
    model.add(layers.Dense(256 * 7 * 7, input_dim=noise_dim))
    model.add(layers.Reshape((7, 7, 256)))
    #卷积层，7*7*256变为14*14*128
    model.add(layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    #反卷积层，14*14*128变为14*14*64
    model.add(layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    #反卷积层，14*14*64变为28*28*1
    model.add(layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
    model.add(layers.Activation('tanh'))
    return model

#判别器
def discriminator(img_shape):
    model = Sequential()
    #从28*28*1变成14*14*32
    model.add(
        layers.Conv2D(32,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same', activation='relu'))
    model.add(layers.Dropout(0.3))
    #从14*14*32到7*7*64
    model.add(
        layers.Conv2D(64,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same', activation='relu'))
    model.add(layers.Dropout(0.3))
    #从7*7*64变成3*3128
    model.add(
        layers.Conv2D(128,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same', activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes))
    return model

def discriminator_supervised(discriminator_net):
    model = Sequential()
    model.add(discriminator_net)
    model.add(layers.Activation('softmax'))
    return model

def discriminator_unsupervised(discriminator_net):
    model = Sequential()
    model.add(discriminator_net)
    
    def predict(x):
        #二元化
        prediction = 1.0 - (1.0 /
                            (K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))
        return prediction
 
    model.add(layers.Lambda(predict))
    return model

def gan_net(generator_net, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

#获得有标签数据、无标签数据和测试数据
x_labeled, y_labeled, x_unlabeled, y_unlabeled, x_test, y_test = \
                    get_data(num_labeled)
#鉴定器网络
discriminator_net = discriminator(img_shape)

#有监督鉴定器
discriminator_supervised = discriminator_supervised(discriminator_net)
discriminator_supervised.compile(loss='categorical_crossentropy',
                                 metrics=['accuracy'],
                                 optimizer=tf.keras.optimizers.Adam())
#无监督鉴定器
discriminator_unsupervised = discriminator_unsupervised(discriminator_net)
discriminator_unsupervised.compile(loss='binary_crossentropy',
                                   optimizer=tf.keras.optimizers.Adam())
#生成器
generator = generator_net(noise_dim)
discriminator_unsupervised.trainable = False
gan = gan_net(generator, discriminator_unsupervised)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam())

supervised_losses = []
iteration_checkpoints = []
def train(epochs, batch_size, sample_interval):
    #真实图像的标签：全为1
    real = np.ones((batch_size, 1))
    #伪图像的标签：全为0
    fake = np.zeros((batch_size, 1))
    for iter in range(epochs):
        #****训练鉴定器
        #获得标签样本
        index = np.random.randint(0, len(x_labeled), size = batch_size)
        imgs, labels = x_labeled[index], y_labeled[index]
        #获得无标签样本
        index = np.random.randint(0, len(x_unlabeled), size = batch_size)
        imgs_unlabeled = x_unlabeled[index]
        #生成一批伪图像
        z = np.random.normal(0, 1, (batch_size, noise_dim))
        gen_imgs = generator.predict(z)
 
        #训练有标签的真实样本
        d_loss_supervised, accuracy = discriminator_supervised.train_on_batch(imgs, labels)
        #训练无标签的真实样本
        d_loss_real = discriminator_unsupervised.train_on_batch(
            imgs_unlabeled, real)
        #训练伪样本
        d_loss_fake = discriminator_unsupervised.train_on_batch(gen_imgs, fake)
        d_loss_unsupervised = 0.5 * np.add(d_loss_real, d_loss_fake)
        #****训练生成器
        #生成一批次伪图像
        z = np.random.normal(0, 1, (batch_size, noise_dim))
        gen_imgs = generator.predict(z)

        g_loss = gan.train_on_batch(z, np.ones((batch_size, 1)))
 
        if (iter + 1) % verbose_interval == 0:
            #保存鉴别器的有监督分类损失
            supervised_losses.append(d_loss_supervised)
            iteration_checkpoints.append(iter + 1)
            #输出训练过程
            print(
                "%d [discriminator supervised loss: %.4f, acc.: %.2f%%] [discriminator unsupervised loss: %.4f] [generator loss: %f]"
                % (iter + 1, d_loss_supervised, 100 * accuracy,
                   d_loss_unsupervised, g_loss))
                    
train(epochs, batch_size, verbose_interval)

losses = np.array(supervised_losses)
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses, label="Discriminator loss")
plt.xticks(iteration_checkpoints, rotation=90)
plt.title("Discriminator  Supervised Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()

#计算无标签集合分类准确率
x, y = x_unlabeled, y_unlabeled
_, accuracy = discriminator_supervised.evaluate(x, y)
print("Training—unlabeled Accuracy: %.2f%%" % (100 * accuracy))

#计算测试集上的精度
x, y = x_test, y_test
_, accuracy = discriminator_supervised.evaluate(x, y)
print("Test Accuracy: %.2f%%" % (100 * accuracy))