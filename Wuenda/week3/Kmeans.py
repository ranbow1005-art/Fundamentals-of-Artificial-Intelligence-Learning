import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from kneed import KneeLocator

#1.肘部法取k值
def find_k_elbow(X):
    sse=[]
    k_range=range(1,10)
    for k in k_range:
        # 建立KMeans模型，且默认采用k-means++初始化策略
        kmeans=KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X)
        #kmeans.inertia_是当前k值对应的sse
        sse.append(kmeans.inertia_)

    #选取最佳k值
    kneedle = KneeLocator(k_range, sse, curve='convex', direction='decreasing')
    optimal_k=kneedle.knee
    #
    # #绘图
    # plt.figure(figsize=(8, 5))
    # plt.plot(k_range, sse, marker='o', linestyle='-', color='b', linewidth=2)
    # plt.title('Elbow Method For Optimal K', fontsize=14)
    # plt.xlabel('Number of clusters (K)', fontsize=12)
    # plt.ylabel('Inertia (SSE / Distortion)', fontsize=12)
    # plt.xticks(k_range)  # 确保 X 轴刻度显示为整数 1 到 9
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.show()
    return optimal_k

#2.使用肘部法的kmeans算法
def run_kmeans(X):
    k=find_k_elbow(X)
    #初始化kmeans对象
    #使用k-means++进行10次初始化，并取效果最好的一次
    kmeans=KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    #拟合数据
    kmeans.fit(X)
    centroids=kmeans.cluster_centers_#最终质心坐标
    labels=kmeans.labels_#样本标签
    sse=kmeans.inertia_
    return centroids, labels, sse

def plot_kmeans_results(X, labels, centroids):
    """绘制 KMeans 聚类后的二维几何空间效果图

    Args:
        X (ndarray): 样本数据集，形状为 (m, 2) —— 绘图默认支持二维数据
        labels (ndarray): 每个样本所属的簇标签，形状为 (m,)
        centroids (ndarray): 最终收敛的 K 个质心坐标，形状为 (K, 2)
    """
    plt.figure(figsize=(8, 6))

    # 1. 绘制所有样本点，并根据算法预测的标签(labels)自动涂上不同的颜色
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="viridis",
        edgecolors="k",
        s=50,
        alpha=0.7,
        label="Data Points",
    )

    # 2. 绘制最终的质心位置（采用醒目的红色大“X”记号）
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        marker="X",
        s=250,
        edgecolors="black",
        linewidths=1.5,
        label="Final Centroids",
    )

    # 3. 图像细节美化
    plt.title("Visualization of K-Means Clustering Results", fontsize=14, pad=15)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)  # 开启半透明网格线
    plt.legend(loc="upper right", fontsize=11)  # 显示图例

    # 4. 渲染呈现
    plt.show()

if __name__ == "__main__":
    # make_blobs产生聚类数据集
    X, y = make_blobs(n_samples=100, centers=3, cluster_std=0.8, random_state=42)
    final_centers, sample_labels, _= run_kmeans(X)
    plot_kmeans_results(X, sample_labels, final_centers)

