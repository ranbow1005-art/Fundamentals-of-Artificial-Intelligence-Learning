import tensorflow as tf
import numpy as np
from keras import models, layers

#1.处理数据
#1.1 加载MNIST数据集
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#1.2 MNIST数据集中图片大小为28*28，需填充至32*32
#(0,0)对应样本数量N，中间的(2,2)对应高度H，上下各填充两个，末尾的(2,2)对应宽度，左右各填充两个
x_train=np.pad(x_train,((0,0),(2,2),(2,2)),'constant')
x_test=np.pad(x_test,((0,0),(2,2),(2,2)),'constant')
#1.3 归一化，并添加通道维度(N,H,W,C)
x_train=x_train.reshape(-1,32,32,1).astype('float32')/255.0
x_test=x_test.reshape(-1,32,32,1).astype('float32')/255.0

#2.构建神经网络架构
model=models.Sequential([
    layers.Conv2D(filters=6,kernel_size=(5,5),strides=1,padding='valid',activation='sigmoid',input_shape=(32,32,1),name='C1'),
    layers.AveragePooling2D(pool_size=(2,2),strides=2,name='S2'),
    layers.Conv2D(filters=16,kernel_size=(5,5),strides=1,padding='valid',activation='sigmoid',name='C3'),
    layers.AveragePooling2D(pool_size=(2,2),strides=2,name='S4'),
    layers.Flatten(name='Flatten'),
    layers.Dense(units=120,activation='sigmoid',name='FC3'),
    layers.Dense(units=84,activation='sigmoid',name='FC4'),
    layers.Dense(units=10,activation='softmax',name='Output'),
])
#输出网络结构摘要,核对每层的张量形状与参数量
#model.summary()

#3.编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#4.训练模型
model.fit(x_train,y_train,epochs=5,batch_size=32,validation_data=(x_test,y_test))

#5.评估模型
test_loss,test_acc=model.evaluate(x_test,y_test)
print(f"测试集损失为:{test_loss:.2f}")
print(f"测试集准确率为:{test_acc:.2f}")