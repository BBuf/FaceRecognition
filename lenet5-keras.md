

```python
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from keras.utils import np_utils

# parameters
patience = 10
log_file_path = "./log.csv"
trained_models_path = "./LeNet-5"

# load dataset
# 训练集为60000张28 * 28像素灰度图像
# 测试集为10000同规格图像，总共10类数字标签
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 增加一个维度并对图片数据归一化
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) / 255.0
# 转换为one-hot标签
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
# model callbacks
early_stop = EarlyStopping('loss', 0.1, patience=patience)
reduce_lr = ReduceLROnPlateau('loss', factor=0.1, patience=int(patience/2), verbose=1)
csv_logger = CSVLogger(log_file_path, append=False)
model_names = trained_models_path + '.{epoch:02d}-{acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, monitor='loss', verbose=1, save_best_only=True, save_weights_only=False)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# LeNet-5
model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test), callbacks=callbacks, verbose=1, shuffle=True)
```

    D:\Anconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 24, 24, 6)         156       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 12, 12, 6)         0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 8, 8, 16)          2416      
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 4, 4, 16)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 120)               30840     
    _________________________________________________________________
    dense_2 (Dense)              (None, 84)                10164     
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                850       
    =================================================================
    Total params: 44,426
    Trainable params: 44,426
    Non-trainable params: 0
    _________________________________________________________________
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/5
    59904/60000 [============================>.] - ETA: 0s - loss: 0.6774 - acc: 0.7734- ETA: 5s - loss: 0.7344 - acEpoch 00001: loss improved from inf to 0.67657, saving model to ./LeNet-5.01-0.77.hdf5
    60000/60000 [==============================] - 56s 934us/step - loss: 0.6766 - acc: 0.7737 - val_loss: 0.1655 - val_acc: 0.9478
    Epoch 2/5
    59904/60000 [============================>.] - ETA: 0s - loss: 0.1430 - acc: 0.9548Epoch 00002: loss improved from 0.67657 to 0.14292, saving model to ./LeNet-5.02-0.95.hdf5
    60000/60000 [==============================] - 57s 957us/step - loss: 0.1429 - acc: 0.9549 - val_loss: 0.0953 - val_acc: 0.9697
    Epoch 3/5
    59904/60000 [============================>.] - ETA: 0s - loss: 0.1030 - acc: 0.9682- ETA: 1s - loss: 0.1031 - Epoch 00003: loss improved from 0.14292 to 0.10294, saving model to ./LeNet-5.03-0.97.hdf5
    60000/60000 [==============================] - 52s 859us/step - loss: 0.1029 - acc: 0.9682 - val_loss: 0.0847 - val_acc: 0.9731
    Epoch 4/5
    59904/60000 [============================>.] - ETA: 0s - loss: 0.0810 - acc: 0.9744- ETA: 2s - Epoch 00004: loss improved from 0.10294 to 0.08102, saving model to ./LeNet-5.04-0.97.hdf5
    60000/60000 [==============================] - 47s 785us/step - loss: 0.0810 - acc: 0.9743 - val_loss: 0.0760 - val_acc: 0.9763
    Epoch 5/5
    59904/60000 [============================>.] - ETA: 0s - loss: 0.0711 - acc: 0.9774- ETA: 7s - loss: 0.0707 - acc: 0.  - - ETA: 0s - loss: 0.0710 - acc: 0.9Epoch 00005: loss improved from 0.08102 to 0.07101, saving model to ./LeNet-5.05-0.98.hdf5
    60000/60000 [==============================] - 48s 800us/step - loss: 0.0710 - acc: 0.9775 - val_loss: 0.0790 - val_acc: 0.9772
    




    <keras.callbacks.History at 0x1c4ca14dbe0>




```python
from keras.utils.vis_utils import plot_model
#plot_model(model, to_file='LeNet-5.png', show_shapes=True, show_layer_names=False)
```


```python
import keras
from keras import backend as K
import numpy as np

def get_feature_function(model_path, output_layer_index):
    # 载入持久化的模型内存中
    model = keras.models.load_model(model_path)
    vector_function = K.function([model.layers[0].input], [model.layers[output_layer_index].output])
    def inner(input_data):
        vector = vector_function([input_data])[0]
        return vector.flatten()
    return inner

import cv2
path = "F:\cat.jpg"

# 载入模型文件
get_feature = get_feature_function(model_path="./LeNet-5.05-0.98.hdf5", output_layer_index=7)
# 读入图片数据
img = cv2.imread(path)
# 修改图片尺寸
img = cv2.resize(img, (28, 28))
# 转换为灰度图像
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 此时img的shape为(28, 28)
# 为img增加两个维度，使其格式与训练时候的输入shape(None,1,28,28)一致
img = np.expand_dims(img, -1)
img = np.expand_dims(img, 0)
# 获取特征向量
feature = get_feature(img)
# 打印特征向量
print(feature)
```

    [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
    
