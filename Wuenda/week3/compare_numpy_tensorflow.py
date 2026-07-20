import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import time

def compare_numpy_tensorflow(np_matrix_a,np_matrix_b):
    tensorflow_matrix_a=tf.convert_to_tensor(np_matrix_a)
    tensorflow_matrix_b=tf.convert_to_tensor(np_matrix_b)
    # 提前运行一次计算，消除首次调用的框架初始化与内存分配开销
    tf.matmul(tensorflow_matrix_a, tensorflow_matrix_b)
    #1.先算numpy操作矩阵耗时
    np_start_time=time.perf_counter()#perf_counter()比time()精度更高
    np.dot(np_matrix_a,np_matrix_b)
    np_end_time=time.perf_counter()
    #2.再算TensorFlow操作矩阵耗时
    tf_start_time=time.perf_counter()
    tf.matmul(tensorflow_matrix_a,tensorflow_matrix_b)#tensorflow中两矩阵相乘的函数
    tf_end_time=time.perf_counter()
    print(f"numpy操作矩阵所用时间为：{np_end_time-np_start_time:.5f}")
    print(f"tensorflow操作矩阵所用时间为：{tf_end_time - tf_start_time:.5f}")

size=5000
#np.random.rand()默认生成64位双精度浮点数，GPU对32位单精度浮点数才有极强的优化
matrix_a=np.random.rand(size,size).astype(np.float32)
matrix_b=np.random.rand(size,size).astype(np.float32)
compare_numpy_tensorflow(matrix_a,matrix_b)