import glob
import  numpy as np
import os
import shutil
np.random.seed(100)
files = glob.glob('train_6_1/*')
#训练集合
cat_images = [image for image in files if 'cat' in image]
dog_images = [image for image in files if 'dog' in image]
cat_train = np.random.choice(cat_images, size=1000, replace=False)
dog_train = np.random.choice(dog_images, size=1000, replace=False)
#验证集合
cat_images = list(set(cat_images)-set(cat_train))
dog_images = list(set(dog_images)-set(dog_train))
cat_val = np.random.choice(cat_images, size=500, replace=False)
dog_val = np.random.choice(dog_images, size=500, replace=False)
#测试集合
cat_images = list(set(cat_images)-set(cat_val))
dog_images = list(set(dog_images)-set(dog_val))
cat_test = np.random.choice(cat_images, size=500, replace=False)
dog_test = np.random.choice(dog_images, size=500, replace=False)
#建立文件夹train、val、test，并拷贝相应数据
train_images = np.concatenate([cat_train, dog_train])
val_images = np.concatenate([cat_val, dog_val])
test_images = np.concatenate([cat_test, dog_test])
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'
os.mkdir(train_dir) if not os.path.isdir(train_dir) else None
os.mkdir(val_dir) if not os.path.isdir(val_dir) else None
os.mkdir(test_dir) if not os.path.isdir(test_dir) else None
for image in train_images:
    shutil.copy(image, train_dir)
for image in val_images:
    shutil.copy(image, val_dir)
for image in test_images:
    shutil.copy(image, test_dir)
