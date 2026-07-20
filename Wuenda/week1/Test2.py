import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')

#x_train=np.array([1.0,2.0])
#y_train=np.array([300.0,500.0])
#plt_intuition(x_train,y_train)

#根据J(w,b)公式计算代价函数
def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost_sum=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost_sum=cost_sum+(f_wb-y[i])**2
    total_cost=(1/(2*m))*cost_sum
    return total_cost

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,]) 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
#soup_bowl()

import math,copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_gradient(x,y,w,b):
    m=x.shape[0]
    #初始化偏导累加变量
    dj_dw=0
    dj_db=0
    for i in range(m):
        #计算当前样本预测值
        f_wb=w*x[i]+b
        #计算当前样本误差值
        error=f_wb-y[i]
        dj_dw+=error*x[i]
        dj_db+=error
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db

def gradient_descent(x,y,w_in,b_in,alpha,num_iters,cost_function,gradient_function):
    #num_iters是运行梯度下降的迭代次数
    w=copy.deepcopy(w_in)#深拷贝，会直接开辟新的内存空间进行复制存储
    #存放每次迭代后的代价函数和w，b，便于画图
    J_history=[]
    P_history=[]
    #w=w_in #课件中出现深拷贝后又重新赋值的情况？
    b=b_in
    for i in range(num_iters):
        #num_iters次迭代更新w和b
        dj_dw,dj_db=gradient_function(x,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        #存储w，b每次更新后得到的J
        if i<100000:#防止资源耗尽
            J_history.append(cost_function(x,y,w,b))
            P_history.append([w,b])
        #根据迭代次数打印十次，若迭代次数不足十次，则打印num_iters次
        if i%math.ceil(num_iters/10)==0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w,b,J_history,P_history

#初始化参数
w_init=0
b_init=0
iterations=10000
tmp_alpha=1.0e-2
#执行gradient_descent
w_final,b_final,J_hist,P_hist=gradient_descent(x_train,y_train,w_init,b_init,tmp_alpha,iterations,compute_cost,compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()