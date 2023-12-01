#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:08:50 2022

@author: dingjc
"""

import random
from imutils import paths
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

#超参
image_dims = (128, 128, 3)
epochs = 30
lr = 0.0001
classes = 6
batch_size = 32

#模型
def net():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same",
            input_shape=image_dims))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization())    
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization())    
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization())    
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.3))
    
    model.add(Dense(classes))
    model.add(Activation('sigmoid'))

    return model

imagePaths = sorted(list(paths.list_images('multi_label')))
random.seed(100)
random.shuffle(imagePaths)

# initialize the data and labels
data = []
labels = []
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_dims[1], image_dims[0]))
    image = img_to_array(image)
    data.append(image)

    # extract set of class labels from the image path and update the
    # labels list
    l = label = imagePath.split(os.path.sep)[-2].split("_")
    labels.append(l)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
mlb = MultiLabelBinarizer()
mlb = mlb.fit(labels)
labels = mlb.transform(labels)

print(mlb.classes_)
print('(blue, dress):', (mlb.transform([('blue', 'dress')])))

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.1, random_state=100)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

model = net()
#model.summary()

opt = Adam(learning_rate=lr, decay=lr/epochs)

model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["binary_accuracy"])

# train the network
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // batch_size,
    epochs=epochs, verbose=1)

fig = plt.figure(1)
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["binary_accuracy"], label="train_accuracy")
plt.plot(np.arange(0, epochs), H.history["val_binary_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.show()

fig = plt.figure(figsize=(10,4))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(testX[i*50,])
    print(model.predict(np.expand_dims(testX[i*50,], axis=0)))
    print(testY[i*50,])
plt.show()


