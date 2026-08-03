import jieba
import numpy as np
import tensorflow as tf
#1准备数据
#1.1创建语录
corpus=[
    "自然语言处理是人工智能领域的重要方向",
    "深度学习为自然语言处理带来了巨大的突破",
    "词嵌入和词向量是自然语言处理的基础技术",
    "神经网络模型可以有效地学习文本的语义表示"
]
#1.2使用jieba进行中文分词
corpus_tokens=[list(jieba.cut(sentence)) for sentence in corpus]
#1.3构建词表与映射
vocab=set()
for tokens in corpus_tokens:
    for token in tokens:
        vocab.add(token)
vocab=list(vocab)
vocab_size=len(vocab)
#1.3.2创建双向映射字典
word2id={word:i for i,word in enumerate(vocab)}
id2word={i:word for i, word in enumerate(vocab)}
#1.4构建Skip-gram训练数据集
"""
CBOW中用上下文获取中心词
Skip-gram用中心词获取上下文
因此X和y存储的内容将发生改变
"""
WINDOW_SIZE=2 #单侧窗口为2
X_data=[]#存储中心词ID
y_data=[]#存储上下文词ID列表
for tokens in corpus_tokens:
    token_ids=[word2id[word] for word in tokens]
    """
    #进行一个小优化，跳过过短的句子，防止滑动窗口越界
    if len(token_ids)<2*WINDOW_SIZE+1:
        continue
    上述优化过程中被跳过的短句将直接无法得到训练，对于小训练集的模型训练而影响较大
    """
    #滑动窗口扫描整句话
    #取左边WINDOW_SIZE个词和右边WINDOW_SIZE个词
    for i in range(WINDOW_SIZE,len(token_ids)-WINDOW_SIZE):
        #输入中心词id
        target=token_ids[i]
        #获取周围WINDOW_SIZE范围内的上下文词索引
        context_indices=list(range(i-WINDOW_SIZE,i))+list(range(i+1,i+WINDOW_SIZE+1))
        #将每一个（中心词，上下文词）拆分为独立的训练样本对
        for j in context_indices:
            X_data.append(target)
            y_data.append(token_ids[j])
"""
此时的X_data和y_data是list类型，需要转化为numpy数组，主要原因如下：
1.原生list没有shape属性，无法让keras知道列表包含的样本数和每个样本有多少特征
2.list为指针数组，在内存中分散存放。numpy数组在内存中连续存放，读取速度更快。
"""
X_data=np.array(X_data)
y_data=np.array(y_data)

#2构建Skip-gram模型
EMBEDDING_DIM=16#词向量维度设定
class SkipGramModel(tf.keras.Model):
    def __init__(self,vocab_size,embedding_dim):
        super().__init__()
        #词嵌入层
        self.embedding=tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="word_embedding"
        )
        #全连接输出层：输出词表中各个词成为上下文词的概率
        self.dense=tf.keras.layers.Dense(
            vocab_size,
            activation="softmax",
        )
    def call(self,inputs):
        # inputs形状：(batch_size,) 即单个中心词的id
        embeds=self.embedding(inputs)# 形状:(batch_size,embedding_dim)
        #核心区别：不再需要用reduce_mean取平均向量
        output=self.dense(embeds)#形状：(batch_size,vocab_size)
        return output
#实例化模型
model=SkipGramModel(vocab_size=vocab_size,embedding_dim=EMBEDDING_DIM)

#3训练模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
#打印
class PrintEveryNEpochs(tf.keras.callbacks.Callback):
    def __init__(self,period):
        super(PrintEveryNEpochs,self).__init__()
        self.period=period#设定打印间隔
    def on_epoch_end(self,epoch,logs=None):
        # epoch从0开始计数
        if (epoch+1) % self.period == 0:
            loss=logs.get('loss')
            accuracy=logs.get('accuracy')
            print(f"Epoch{epoch + 1}/{self.params['epochs']}-loss: {loss:.4f}, accuracy: {accuracy:.4f}")

history=model.fit(X_data,y_data,epochs=500,
                  verbose=0,callbacks=[PrintEveryNEpochs(50)])
print(f"训练完成，最终Loss={history.history['loss'][-1]:.4f}")

#4提取训练好的词向量并测试
#提取Embedding层的权重矩阵，形状为(vocab_size,embedding_dim)
word_vectors=model.embedding.get_weights()[0]
def get_word_vector(word):
    #根据词语获取对应的词向量
    if word in word2id:
        word_id=word2id[word]
        return word_vectors[word_id]
    else:
        return None
def cosine_similarity(word_vector1,word_vector2):
    #计算两个向量的余弦相似度,数值越接近1表示语义越相似
    return np.dot(word_vector1,word_vector2)/(np.linalg.norm(word_vector1)*np.linalg.norm(word_vector2))
#测试
test_word_1="自然语言"
test_word_2="人工智能"
vec1=get_word_vector(test_word_1)
vec2=get_word_vector(test_word_2)
if vec1 is not None and vec2 is not None:
    similarity=cosine_similarity(vec1,vec2)
    print(f"{test_word_1}的词向量是{vec1}")
    print(f"'{test_word_1}'与'{test_word_2}'的余弦相似度:{similarity:.4f}")
else:
    print("test_word_1或test_word_2输入有误")



