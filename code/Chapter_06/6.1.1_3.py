from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import  Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.preprocessing import LabelEncoder

IMAGE_DIMS = (256, 256, 3)
def get_data():
    train_files = glob.glob('train/*')
    train_images = [img_to_array(load_img(img, target_size=IMAGE_DIMS)) for img in train_files]
    train_images = np.array(train_images)
    train_images = train_images/255.0
    train_labels = [file.split('.')[0][-3:].strip() for file in train_files]
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    val_files = glob.glob('val/*')

    val_images = [img_to_array(load_img(img, target_size=IMAGE_DIMS)) for img in val_files]
    val_images = np.array(val_images)
    val_images = val_images/255.0
    val_labels = [file.split('.')[0][-3:].strip()  for file in val_files]
    val_labels = le.fit_transform(val_labels)
    return train_images, train_labels, val_images, val_labels
#获得数据
x, y, val_x, val_y = get_data()
#vgg模型，weights='imagenet'
vgg = VGG16(include_top=False, weights='imagenet', input_shape=IMAGE_DIMS)
output = vgg.layers[-1].output
output = Flatten()(output)
vgg_model = Model(vgg.input, output)
#设置vgg网络参数不可训练（冰冻）
for layer in vgg_model.layers:
    layer.trainable = False
vgg_model.summary()
#搭建模型，前端为vgg（参数冰冻），后端为定制层
model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['acc'])
history = model.fit(x=x, y=y, validation_data=(val_x, val_y), batch_size=32, epochs=10, verbose=1)