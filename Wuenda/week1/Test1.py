import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')#图表样式
x_train=np.array([1.0,2.0])
y_train=np.array([300.0,500.0])
# 绘制图表
#原始数据点,marker形状，c颜色
plt.scatter(x_train,y_train,marker='x',c='r',label='Actual Values')
#表名
plt.title("Housing Prices")
#y轴标签
plt.ylabel("Price (in 1000s of dollars)")
#x轴标签
plt.xlabel("Size (1000 sqft)")

#参数设置
w=200
b=100
m=x_train.shape[0]#获取x_train中元素的个数
f_wb=np.zeros(m)#np.zeros(n)将返回一个带有n个元素的一维numpy数组，内存申请
for i in range(m):
    f_wb[i]=w*x_train[i]+b

#绘制函数结果
plt.plot(x_train,f_wb,c='b',label='Our Prediction')#plt.plot用直线连接离散点
plt.legend()
plt.show()

x=1.2
cost_1200sqft=w*x+b
print(f"1200平方英尺价格为{cost_1200sqft}元")