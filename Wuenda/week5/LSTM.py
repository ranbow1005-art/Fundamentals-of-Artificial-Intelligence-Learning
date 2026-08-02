import numpy as np
from keras import Sequential, layers
from matplotlib import pyplot as plt

#1.准备数据
#1.1生成正弦波数据
time=np.linspace(0,1,2000)#生成两千个时间点的数据
data=np.sin(time)
#1.2构建滑动窗口数据集
""""
为什么要构建这个滑动窗口数据？
首先，这个滑动窗口数据是将正弦波以滑动窗口拆分，窗口中的前50项为x，第51项为真实标签y
因此，一个长度为len(data)的序列数据可以进行len(data)-seq_length次窗口滑动
相较于直接将1000个数据简单划分为50个为一组最终只能得到20组数据相比更有助于模型的训练
此外，滑动窗口取得的数据连续性更好，更易学习到正弦波中的规律
"""
#用前seq_length个时间步的数据，预测接下来的1个时间步
def create_dataset(data,seq_length=50):
    X,y=[],[]
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X),np.array(y)
#1.3划分正弦波数据，并将x分为训练集，验证集和测试集70%/15%/15%
seq_length=50
X,y=create_dataset(data,seq_length)
X_size=X.shape[0]
X_train,y_train=X[:int(X_size*0.7)],y[:int(X_size*0.7)]
X_val,y_val=X[int(X_size*0.7):int(X_size*0.85)],y[int(X_size*0.7):int(X_size*0.85)]
X_test,y_test=X[int(X_size*0.85):],y[int(X_size*0.85):]
#2.构建模型
model=Sequential([
    layers.LSTM(units=64,activation='tanh',input_shape=(seq_length,1)),
    layers.Dense(units=32,activation='relu'),#补充模型非线性拟合能力
    layers.Dense(units=1)#回归预测，不作说明默认Linear
])
#3.编译模型
model.compile(optimizer='adam',loss='mse')
model.summary()
#4.训练模型
history=model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_val,y_val),
)
#5.评估
test_loss=model.evaluate(X_test, y_test)
print(f"测试集Loss: {test_loss:.6f}")
predictions = model.predict(X_test)
# 6. 结果可视化
plt.figure(figsize=(14, 5))
# 绘制 Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
# 绘制测试集预测对比曲线
plt.subplot(1, 2, 2)
plt.plot(y_test, label='True Sin Wave (Test)', color='blue')
plt.plot(predictions, label='LSTM Prediction', color='red', linestyle='--')
plt.title('Test Set Prediction Performance')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()