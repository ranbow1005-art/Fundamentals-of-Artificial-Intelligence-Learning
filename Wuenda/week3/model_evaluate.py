import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

#1.准备数据集，模拟生成100房屋的size和price
np.random.seed(42)
#房子尺寸在1000-3000
X=np.random.randint(1000,3000,size=(100,1))
y=0.15*X.squeeze()+50+np.random.normal(0,30,size=100)#squeeze()将(100,1)的二维压成(100,)的一维

#2.将数据集70%设置为训练集，30%设置为测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#3.通过高阶多项式训练模型
degree=10
model=make_pipeline(StandardScaler(),  #把1000~3000缩放到小范围，防止计算 X^10 时数值溢出
    PolynomialFeatures(degree=degree, include_bias=False),  #生成高阶多项式特征
    LinearRegression(),)
model.fit(X_train,y_train)

#4.评估模型
def compute_error(X_data,y_data,model):
    m = len(y_data)
    predictions = model.predict(X_data)
    # 代价函数计算
    sum_squared_error = np.sum((predictions - y_data) ** 2)
    error = (1 / (2 * m)) * sum_squared_error
    return error

# 5.计算并打印结果
j_train=compute_error(X_train,y_train,model)
j_test=compute_error(X_test,y_test,model)
print("训练集误差：",j_train)
print("测试集误差：",j_test)

