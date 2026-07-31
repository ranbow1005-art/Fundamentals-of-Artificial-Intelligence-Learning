import keras
from keras import models, layers
import numpy as np

#识别MNIST手写数字
#1.数据预处理
#1.1获取样本
mnist=keras.datasets.mnist#获取keras框架中内置的MNIST手写数字数据集模块
#在默认的MNIST数据集中，一共有70000张手写数字图片
#训练集(X_train, y_train)有60000个样本，测试集(X_test, y_test)有10000个样本
#mnist.load_data()返回值为包含两个元素的嵌套元组((X_train, y_train),(X_test, y_test))，而非包含4个元素的扁平元组
(X_train, y_train), (X_test, y_test)=mnist.load_data()
#1.2对像素值进行归一化(0~255 -> 0.0~1.0)
X_train=X_train/255.0
X_test=X_test/255.0
#1.3将图像数据(N,28,28)转变为一维向量(N,784) 28*28=784 N表示样本数量
X_train=X_train.reshape(-1,784)#-1为占位符，计算机将根据(数据集总数值)/(每个样本的数值)得到样本数量
X_test=X_test.reshape(-1,784)

#2.创建神经网络结构
model=models.Sequential([
    layers.Dense(25,activation="relu",input_shape=(784,)),#第一层要告知模型输入数据的维度
    layers.Dense(15,activation="relu"),
    layers.Dense(10,activation=None),
])

#3.选择模型和代价函数
model.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])#准确率评估指标

#4.训练模型
model.fit(X_train,y_train,epochs=5,batch_size=32,validation_split=0.2)

#5.模型评估
test_loss,test_acc=model.evaluate(X_test,y_test)
print(f"最终损失:{test_loss:.2f}")
print(f"最终准确率:{test_acc*100:.2f}%")

#6.随机取测试集中一个样本进行测试
index=np.random.randint(0,len(X_test))
sample_image=X_test[index:index+1]
true_label=y_test[index]
pred_label=model.predict(sample_image)
print(f"样本实际值为：{true_label},模型预测值为:{np.argmax(pred_label)}")