# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers
import os

# %%
x = tf.random.normal([2, 784])
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(x, w1) + b1  #
o1 = tf.nn.relu(o1)
o1

# %%
x = tf.random.normal([4, 4])
# 创建全连接层，指定输出节点数和激活函数
fc = layers.Dense(3, activation=tf.nn.relu)
h1 = fc(x)  # 通过fc类完成一次全连接层的计算

# %%
fc

#%%
fc.kernel

# %%
fc.bias

# %%
fc.non_trainable_variables

# %%
fc.trainable_variables

# %%
# 对于全连接层，内部张量都参与梯度优化，故 variables 返回列表与 trainable_variables 一 样。
fc.variables

# %%
vars(fc)

# %%
embedding = layers.Embedding(10000, 100)

# %%
x = tf.ones([25000, 80])

# %%

embedding(x)

# %%
'''在 Softmax 函数的数值计算过程中，容易因输入值偏大发生数值溢出现象；
在计算交叉熵时，也会出现数值溢出的问题。为了数值计算的稳定性，
TensorFlow 中提供了一个统一的接口，将 Softmax 与交叉熵损失函数同时实现，
同时也处理了数值不稳定的异常，一般推荐使用，避免单独使用 Softmax 函数与交叉熵损失函数。  
'''

'''
y_true 代表了 one-hot 编码后的真实标签，y_pred 表示网络的预测值，
当 from_logits 设置为 True 时， y_pred 表示须为  未经过  Softmax 函数的变量 z；
当 from_logits 设置为 False 时，y_pred 表示为  经过  Softmax 函数的输出。
'''
z = tf.random.normal([2, 10])  # 构造输出层的输出
y_onehot = tf.constant([1, 3])  # 构造真实值
y_onehot = tf.one_hot(y_onehot, depth=10)  # one-hot编码
# 输出层未使用Softmax函数，故from_logits设置为True
loss = keras.losses.categorical_crossentropy(y_onehot, z, from_logits=True)
loss = tf.reduce_mean(loss)  # 计算平均交叉熵损失
loss


# %%
'''
新建这个类的实例，实现 Softmax 与交叉熵损失函数的计算
'''
criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
loss = criteon(y_onehot, z)  # 计算损失
loss

# %%
# MSE
o = tf.random.normal([2,10]) # 构造网络输出
y_onehot = tf.constant([1,3]) # 构造真实值
y_onehot = tf.one_hot(y_onehot, depth=10)
loss = tf.keras.losses.MSE(y_onehot, o) # 计算均方差
loss

# %%
loss = tf.reduce_mean(loss) # 计算 batch 均方差
loss

# %%
# 也可以通过类方式实现，对应的类为 keras.losses.MeanSquaredError()：
# 创建 MSE 类
criteon = tf.keras.losses.MeanSquaredError()
loss = criteon(y_onehot,o) # 计算 batch 均方差
loss