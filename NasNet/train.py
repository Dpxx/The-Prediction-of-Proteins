import os, cv2, tqdm
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.applications import NASNetMobile
from keras.metrics import binary_crossentropy

from sklearn.model_selection import train_test_split

train_data = os.listdir('imgtrain')
np.random.shuffle(train_data)
test_data = os.listdir('imgtest')
np.random.shuffle(test_data)
train = train_data
test = test_data


def get_data_train(data, batch_size):
    np.random.shuffle(train)
    while True:
        for i in range(int(len(data) // batch_size)):
            x, y = [], []
            for j in data[i * batch_size: (i + 1) * batch_size]:
                # 1,2 [0, 1, 1, 0]
                temp = np.zeros(shape=(10,))
                for k in j.split('.')[0].split('_')[1].split(';'):
                    temp[int(k)] = 1
                img = np.array(cv2.resize(cv2.imread('imgtrain\{}'.format(j)), (224, 224)), dtype=np.float) / 255.0
                x.append(img)
                y.append(temp)
            yield (np.array(x), np.array(y))


def get_data_test(data, batch_size):
    np.random.shuffle(test)
    while True:
        for i in range(int(len(data) // batch_size)):
            x, y = [], []
            for j in data[i * batch_size: (i + 1) * batch_size]:
                # 1,2 [0, 1, 1, 0]
                temp = np.zeros(shape=(10,))
                for k in j.split('.')[0].split('_')[1].split(';'):
                    temp[int(k)] = 1
                img = np.array(cv2.resize(cv2.imread('imgtest\{}'.format(j)), (224, 224)), dtype=np.float) / 255.0
                x.append(img)
                y.append(temp)
            yield (np.array(x), np.array(y))


inputs = Input(shape=(224, 224, 3))

# 搭建NasNetMobile模型，并添加一个全连接层进行分类
nasnet = NASNetMobile(include_top=False,
                      weights='NASNet-mobile-no-top.h5',
                      input_tensor=inputs, pooling='max', input_shape=(224, 224, 3))
net = Dense(10, activation='sigmoid')(nasnet.layers[-1].output)

model = Model(inputs=inputs, output=net)

# 打印模型结构参数
model.summary()

# 编译模型，优化器采用Adam，损失函数采用的是交叉熵
model.compile(optimizer=Adam(), loss=binary_crossentropy, metrics=['accuracy'])

# 训练模型
model.fit_generator(get_data_train(train, batch_size=64), validation_data=get_data_test(test, batch_size=64), steps_per_epoch=len(train) // 64,
                    validation_steps=len(test) // 64,
                    verbose=1, epochs=50, callbacks=[
        CSVLogger('train.log'),
        ModelCheckpoint('model.h5', monitor='val_acc', save_best_only=True, mode='auto', verbose=1),
    ])
# 保存模型
model.save('final_model.h5')
