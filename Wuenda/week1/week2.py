import numpy as np
import time

#向量的创建
a=np.zeros(4)
print(f"np.zeros(4):a={a},a shape={a.shape},a data type={a.dtype}")
a=np.zeros((4,))#创造多维数组的写法，如a=np.zeros((3,4))将会创建一个三行四列的二维数组
print(f"np.zeros((4,)):a={a},a shape={a.shape},a data type={a.dtype}")
a=np.random.random_sample(4)#随机生成4个[0.0, 1.0)之间的浮点数元素
print(f"np.random.random_sample(4):a={a},a shape={a.shape},a data type={a.dtype}")
#综上试验，np.zeros(4)将为向量a分配4个元素类型大小的内存空间，并初始化为0，而且默认为浮点类型.
#a.shape获取向量a的维度，此处是一维，并且在这个维度上有4个元素。
#a.dtype输出a中元素的类型。
a = np.arange(10)#创建从0-9的数组
print(f"a[-1]={a[-1]}")#当index为负时，代表从数组末尾开始计数，但依旧无法超过数组大小
#数组元素切割
a = np.arange(10)
print(f"a={a}")
c=a[2:7:1]#a[起始索引:结束索引:步长]
print(f"a[2:7:1]={c}")
c=a[2:7:2]
print(f"a[2:7:2]={c}")
#单个向量运算
a=np.array([1,2,3,4])
print(f"a:{a}")
#对a中元素取负值
b=-a
print(f"b:{b}")
#求a中所有元素和
b=np.sum(a)
print(f"b:{b}")
#求a中元素的平均值
b=np.mean(a)
print(f"b:{b}")
#对a中所有元素取立方
b=a**3
print(f"b:{b}")
#相同大小向量间的对应元素计算
a=np.array([1,2,3,4])
b=np.array([-1,-2,3,4])
c=np.array([5,6,-7,-8])
d=a+b+c#向量相加
print(f"a+b+c={d}")
#标量向量运算
e=a*5
print(f"e=a*5:{e}")
#numpy向量点乘公式
a=np.array([1,2,3,4])
b=np.array([-1,-2,3,4])
c=np.dot(a,b)
print(f"np.dot(a,b):{c}")

import copy,math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)#干嘛用的

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

#代价函数
def compute_cost(X,y,w,b):
    m=X.shape[0]
    cost=0.0
    for i in range(m):
        f_wb_i=np.dot(X[i],w)+b
        cost+=(f_wb_i-y[i])**2
    cost=cost/(2*m)
    return cost
#预测函数
def predict(x,w,b):
    p=np.dot(x,w)+b
    return p
#梯度下降偏导
def compute_gradient(X,y,w,b):
    m,n=X.shape
    dj_dw=np.zeros(n)
    dj_db=0.0
    for i in range(m):
        error=predict(X[i],w,b)-y[i]
        for j in range(n):
            dj_dw[j]=dj_dw[j]+error*X[i,j]
        dj_db=dj_db+error
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db
#梯度下降 
def gradient_descent(X,y,w_in,b_in,cost_function,gradient_function,alpha,num_iters):
    w=copy.deepcopy(w_in)
    b=b_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b)
        w-=alpha*dj_dw
        b-=alpha*dj_db
    return w,b
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final= gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

#只有一个特征值x时
x=np.arange(0,20,1)#生成包含20个元素的一维数组[0,1,2,……,19]
y=x**2+1
X=x.reshape(-1,1)#将一维数组x转换为二维列矩阵X（从20维向量变为20行1列）
model_w,model_b=run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-2)
plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.title("no feature engineering")
plt.plot(x,X@model_w + model_b, label="Predicted Value")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
#重构特征
x=np.arange(0,20,1)
y=x**2+1
X=x**2#X取x的平方
X=X.reshape(-1,1)#同上，将一维数组转换成二维列矩阵（20行1列）
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

