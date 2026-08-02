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
#1.3构建词表与映射（神经网络无法直接计算字符串，只能处理数字）
#1.3.1收集不重复的词
vocab=set()
for tokens in corpus_tokens:
    for token in tokens:
        vocab.add(token)
vocab=list(vocab)
vocab_size=len(vocab)
#1.3.2创建双向映射字典
"""
enumerate(list)将自动给列表里的每个元素加上索引
比如把['自然','语言']变成(0,'自然')和(1,'语言')----（i，word）
{word:i for i,word in enumerate(vocab)}：
循环遍历每一对（i，word），以word为key，以i为value构建字典
"""
word2id={word:i for i,word in enumerate(vocab)}
id2word={i:word for i, word in enumerate(vocab)}
#1.4构建CBOW训练数据集
WINDOW_SIZE=2 #单侧窗口为2
X_data=[]#存储上下文词ID列表
y_data=[]#存储中心词ID
for tokens in corpus_tokens:
    token_ids=[word2id[word] for word in tokens]
    #滑动窗口扫描整句话
    # 取左边WINDOW_SIZE个词和右边WINDOW_SIZE个词
    for i in range(WINDOW_SIZE,len(token_ids)-WINDOW_SIZE):
        context=token_ids[i-WINDOW_SIZE:i]+token_ids[i+1:i+WINDOW_SIZE+1]
        target=token_ids[i]
        X_data.append(context)
        y_data.append(target)
X_data=np.array(X_data)
y_data=np.array(y_data)

#2构建CBOW模型
"""
CBOW分为输入层隐藏层和输出层
代码实现中隐藏层分为嵌入层(Embedding)和平均层(Mean Pooling)
嵌入层：将输入的上下文词ID转换为高维连续向量
平均层：将所有上下文词向量求平均
"""
EMBEDDING_DIM=16#词向量维度设定
class CBOWModel(tf.keras.Model):
    def __init__(self,vocab_size,embedding_dim):
        super().__init__()
        #词嵌入层
        self.embedding=tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="word_embedding"
        )
        #全连接输出层：维度为词表大小，最后经softmax输出各个词的预测概率
        self.dense=tf.keras.layers.Dense(
            vocab_size,
            activation="softmax",
        )
    def call(self,inputs):
        # inputs形状：(batch_size,context_length) batch_size代表一次训练几句话
        embeds=self.embedding(inputs)# 形状:(batch_size,context_length,embedding_dim)
        #对上下文取平均，再将多个上下文词向量融合成一个单一向量(batch_size,embedding_dim)
        #reduce_mean()将求所有向量的综合平均向量
        mean_embeds=tf.reduce_mean(embeds,axis=1)
        #计算概率
        output=self.dense(mean_embeds)
        return output
#实例化模型
model=CBOWModel(vocab_size=vocab_size,embedding_dim=EMBEDDING_DIM)

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
"""
训练CBOW模型的最终目的不是为了预测中心词
而是为了获取模型内部Embedding层的权重，也就是词向量。
"""
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



