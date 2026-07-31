from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy


#普通方式，容易发生数值上溢
#1.构建神经网络结构
model=Sequential([
    Dense(25,activation="relu"),
    Dense(15,activation="relu"),
    Dense(10,activation="softmax"),
])
#2.选择模型和代价函数
#SparseCategoricalCrossentropy()是稀疏类别交叉熵函数，稀疏表示只能取一个值
model.compile(loss=SparseCategoricalCrossentropy())
#3.训练模型
#model.fit(X,Y,epochs=100)
