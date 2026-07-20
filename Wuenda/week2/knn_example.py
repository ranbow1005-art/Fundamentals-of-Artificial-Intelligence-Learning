from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def knn_iris_example():
    # 1.获取数据集
    iris = load_iris()

    # 2.数据基本处理
    # 数据分割
    # test_size设定测试集占20%，则训练集占80%
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2,
                                                        random_state=22)

    # 3.特征工程
    # 将特征标准化:花萼长度取值4.3-7.9，花瓣宽度取值0.1-2.5,差异较大
    # 对每一个样本特征进行Z-score标准化：x_new=(x-特征均值)/标准差
    transfer = StandardScaler()
    #fit_transform()算出x中每个特征的均值和标准差后再标准化
    X_train = transfer.fit_transform(X_train)
    #transform()根据训练集的均值和标准差进行标准化
    X_test = transfer.transform(X_test)

    #4.模型训练
    #初始化一个KNeighborsClassifier实例，k值为5，算法系统默认选择最优
    #algorithm属性有auto/ball_tree/kd_tree/brute
    estimator = KNeighborsClassifier(n_neighbors=5,algorithm='auto')
    estimator.fit(X_train, y_train)

    #5.模型评估
    score=estimator.score(X_test, y_test)
    print(f"score={score}\n")
    y_pred = estimator.predict(X_test)
    print(f"模型预测值={y_pred}\n")
    print(f"模型真实值={y_test}\n")
    print("对比结果：\n", y_test == y_pred)

#knn_iris_example()

def knn_iris_gscv():
    #1.获取数据集
    iris = load_iris()
    #2.数据基本处理
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2,
                                                        random_state=22)
    #3.特征工程
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    #4.模型训练
    estimator = KNeighborsClassifier()#此处只需要创建实例，不用设置k值，后面GridSearchCV中试k值
    #超参数设定
    param_dict={'n_neighbors':[1,3,5,7]}
    estimator=GridSearchCV(estimator=estimator, param_grid=param_dict,cv=4)#cv代表4折交叉
    estimator.fit(X_train, y_train)

    #5.模型评估
    score=estimator.score(X_test, y_test)#此处是用通过交叉验证和网格搜索得到的最佳k值进行测试集验证后的分数
    print(f"score={score}")
    y_pred = estimator.predict(X_test)
    print(f"模型预测值={y_pred}")
    print(f"模型真实值={y_test}")
    print("对比结果：", y_test == y_pred)

    print(f"\n最佳k值：{estimator.best_params_}")
    print(f"最佳k值对应分数：{estimator.best_score_}")#此处是最佳k值在训练集内部的训练集+验证集训练模式下的平均分数
    print(f"最佳k值对应估计器：{estimator.best_estimator_}")

knn_iris_gscv()