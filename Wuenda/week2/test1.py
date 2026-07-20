import numpy as np
import matplotlib.pyplot as plt

#1.构造模拟数据y=3x+4+噪声
np.random.seed(42)#保证代码的可重复性
X=2*np.random.rand(100,1)#rand取0-1间的随机浮点数,数据在 0 到 1 之间是均等散落的,抽到0.1的概率和抽到0.9的概率一模一样。
y=3*X+4+np.random.randn(100,1)#randn取值概率符合正态分布。绝大多数时候噪声都很小，数据会围绕在0周围。

#小批量SGD实现
def mini_batch_sgd(X,y,learning_rate=0.05,epochs=100,batch_size=16):
    #初始化
    m=X.shape[0]
    w=0.0
    b=0.0
    loss_history=[]

    num_batches=int(np.ceil(m/batch_size))

    for i in range(epochs):
        indices=np.random.permutation(m)#打乱顺序
        for j in range(num_batches):
            #一组一组截取样本
            start=j*batch_size
            end=start+batch_size
            X_batch=X[indices[start:end]]
            y_batch=y[indices[start:end]]
            #1.当前小批次的预测值
            y_pred=X_batch*w+b
            #2.误差
            error=y_pred-y_batch
            #3.计算均方误差的平均梯度
            grad_w=np.mean(2*error*X_batch)
            grad_b=np.mean(2*error)
            #4.更新参数
            w=w-learning_rate*grad_w
            b=b-learning_rate*grad_b
        #每个epoch结束，记录loss，便于观察收敛情况
        loss=np.mean((w*X+b-y)**2)
        loss_history.append(loss)
        #每10轮打印一次
        if (i + 1) % 10 == 0:
            print(f"Epoch {i+1:03d}/{epochs} | Loss: {loss:.4f} | w: {w:.4f}, b: {b:.4f}")
    return w,b,loss_history

print("开始训练：\n")
w_final, b_final, losses = mini_batch_sgd(X, y, learning_rate=0.05, epochs=100, batch_size=16)
print("\n训练结束，验证结果对比：")
print(f"实际目标（带噪声）：w 应接近 3.0, b 应接近 4.0")
print(f"模型学习后：w = {w_final:.4f}, b = {b_final:.4f}")

#可视化
plt.figure(figsize=(12, 5))

# 图1：数据散点图与最终拟合出的直线
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', alpha=0.6, label='True Data (with noise)')
X_line = np.array([[0], [2]])  # 提取两个端点画线
y_line = w_final * X_line + b_final
plt.plot(X_line, y_line, color='red', linewidth=3, label=f'Your SGD Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Data and Fitted Line')
plt.legend()

# 图2：Loss（均方误差）随轮次的下降曲线
plt.subplot(1, 2, 2)
plt.plot(losses, color='orange', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Loss Curve (Must decrease)')

plt.tight_layout()
plt.show()



