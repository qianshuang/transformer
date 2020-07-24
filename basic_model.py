# -*- coding: utf-8 -*-

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64      # 词向量维度
    seq_length = 1000        # 序列长度
    num_classes = 10        # 类别数
    num_filters = 128        # 卷积核数目
    kernel_size = 3         # 卷积核尺寸

    hidden_dim = 128        # 全连接层神经元

    keep_prob = 0.5 # dropout保留比例
    att_keep_prob = 0.9 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 64         # 每批训练大小
    num_epochs = 10         # 总迭代轮次

    print_per_batch = 10    # 每多少轮输出一次结果

    num_heads = 8           # Attention头数
    embeddedPosition = []


class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        # self.embeddedPosition = tf.placeholder(tf.float32, [None, self.config.seq_length, self.config.seq_length], name="embeddedPosition")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())
        self.att_keep_prob = tf.placeholder_with_default(1.0, shape=())  # multi-head attention中的dropout

        self.cnn()

    def _layerNormalization(self, inputs):
        inputsShape = inputs.get_shape() # [batch_size, sequence_length, embedding_size]
        paramsShape = inputsShape[-1:] # 值为embedding_size

        # LayerNorm是在最后的维度上计算输入的数据的均值和方差，BN层是考虑所有维度的
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True) # [batch_size, sequence_len, 1]
        beta = tf.Variable(tf.zeros(paramsShape))
        gamma = tf.Variable(tf.ones(paramsShape))

        epsilon = 1e-8 # 平滑因子，防止除数为0
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta
        return outputs

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            # 模型自己学出词向量矩阵
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            embedding_inputs = tf.concat([embedding_inputs, self.config.embeddedPosition], -1) # position embedding

        with tf.name_scope("cnn"):
            # 1. multi-head attention
            Q = tf.layers.dense(embedding_inputs, self.config.embedding_dim + self.config.seq_length, activation=tf.nn.relu) # [batch_size, sequence_length, embedding_size]
            K = tf.layers.dense(embedding_inputs, self.config.embedding_dim + self.config.seq_length, activation=tf.nn.relu) # tf.layers.dense可以做多维tensor数据的非线性映射
            V = tf.layers.dense(embedding_inputs, self.config.embedding_dim + self.config.seq_length, activation=tf.nn.relu)

            # 将数据按最后一维分割成num_heads个，然后按第一维做拼接
            Q_ = tf.concat(tf.split(Q, self.config.num_heads, axis=-1), axis=0) # [batch_size * numHeads, sequence_length, embedding_size/numHeads]
            K_ = tf.concat(tf.split(K, self.config.num_heads, axis=-1), axis=0)
            V_ = tf.concat(tf.split(V, self.config.num_heads, axis=-1), axis=0)

            # 计算Attention weight
            similary = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            scaledSimilary = similary / (K_.get_shape().as_list()[-1] ** 0.5) # 对计算的点积进行缩放处理，除以向量长度的根号值

            # mask掉Padding的部分：即当该词是Padding时，position embedding也必须为全0
            keyMasks = tf.tile(self.input_x, [self.config.num_heads, 1])
            keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(embedding_inputs)[1], 1])
            paddings = tf.ones_like(scaledSimilary) * (-2 ** (32 + 1))
            maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings, scaledSimilary)

            weights = tf.nn.softmax(maskedSimilary)
            outputs = tf.matmul(weights, V_) # [batch_size * numHeads, sequence_length, embedding_size/numHeads]
            outputs = tf.concat(tf.split(outputs, self.config.num_heads, axis=0), axis=2) # 将多头Attention计算的得到的输出重组成最初的维度[batch_size, sequence_length, embedding_size]
            outputs = tf.nn.dropout(outputs, keep_prob=self.att_keep_prob)

            # 残差网络
            outputs += embedding_inputs

            # layerNormalization
            outputs_1 = self._layerNormalization(outputs) # [batch_size, sequence_length, embedding_size]

            # 2. feed forward
            params = {"inputs": outputs_1, "filters": 128, "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            params = {"inputs": outputs, "filters": self.config.embedding_dim + self.config.seq_length, "kernel_size": 1, "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params) # [batch_size, sequence_length, embedding_size]

            # 残差连接
            outputs += outputs_1

            # layerNormalization
            outputs = self._layerNormalization(outputs) # [batch_size, sequence_length, embedding_size]

            # 全连接层做分类
            outputs = tf.reshape(outputs, [-1, self.config.seq_length * (self.config.embedding_dim + self.config.seq_length)])

        with tf.name_scope("score"):
            fc = tf.contrib.layers.dropout(outputs, self.keep_prob)
            # 激活函数
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(self.logits, 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
