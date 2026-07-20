import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

#1.准备数据集(模拟100套房子的size和price)
np.random.seed(42)
X=np.random.randint(1000,3000,size=(100,1))
y=0.15*X.squeeze()+0.0001*(X.squeeze())**2+50+np.random.normal(0,20,size=100)

#2.划分数据集为训练集、验证集、测试集
#2.1先分20%为测试集
X_rest,X_test,y_rest,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#2.2再分总量的20%为验证集，也就是rest中的25%
X_train,X_validation,y_train,y_validation=train_test_split(X_rest,y_rest,test_size=0.25,random_state=42)

#3.网格搜索测验lambda值
#3.1设置lambda值
lambdas=[0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.0]
#3.2依次对10阶多项式模型进行测试
best_lamb=None
lowest_j_cv=float("inf")
j_train_list = []
j_cv_list = []
for lamb in lambdas:
    model=make_pipeline(StandardScaler(),#第一次缩放,保护高次项计算
                        PolynomialFeatures(degree=10,include_bias=False),
                        StandardScaler(),# 第二次缩放,确保所有特征一起接受λ的惩罚
                        Ridge(alpha=lamb)
                        )
    model.fit(X_train,y_train)
    j_train=mean_squared_error(y_train,model.predict(X_train))/2
    j_cv=mean_squared_error(y_validation,model.predict(X_validation))/2
    j_train_list.append(j_train)
    j_cv_list.append(j_cv)
    if j_cv<lowest_j_cv:
        lowest_j_cv=j_cv
        best_lamb=lamb

#4.输出与绘图
print("最佳lambda值为：",best_lamb)

plt.figure(figsize=(10, 6), dpi=100)

# 将 lambda 转换为字符串列表，作为横轴标签，防止数据点在右侧严重稀疏
x_labels = [str(lamb) for lamb in lambdas]

# 绘制两条核心曲线

plt.plot(
    x_labels,
    j_train_list,
    marker="o",
    linestyle="-",
    color="#1f77b4",
    linewidth=2,
    label="$J_{train}$ (Training Error)",
)
plt.plot(
    x_labels,
    j_cv_list,
    marker="s",
    linestyle="-",
    color="#2ca02c",
    linewidth=2,
    label="$J_{cv}$ (Validation Error)",
)

# 装饰器：让图表更加美观和易读
plt.title(
    "Bias-Variance Tradeoff: Error vs. Regularization $\lambda$",
    fontsize=14,
    pad=15,
)
plt.xlabel("Regularization Parameter $\lambda$ (alpha)", fontsize=12)
plt.ylabel("Cost / Error ($J$)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=12)

# 自动调整布局并展示
plt.tight_layout()
plt.show()