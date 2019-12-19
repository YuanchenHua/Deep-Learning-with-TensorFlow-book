#%% 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential,losses,optimizers,datasets
# %%
class MyDense(layers.Layer):
# 自定义网络层
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        # 创建权值张量并添加到类管理列表中，设置为需要优化
        self.kernel = self.add_weight('w', [inp_dim, outp_dim],trainable=True)
    def call(self, inputs, training=None):
        # 实现自定义类的前向计算逻辑
        # X@W
        out = inputs @ self.kernel
        # 执行激活函数运算
        out = tf.nn.relu(out)
        return out

net = MyDense(4,3)
net.variables,net.trainable_variables

# %%

network = Sequential([MyDense(784, 256), # 使用自定义的层
                MyDense(256, 128),
                MyDense(128, 64),
                MyDense(64, 32),
                MyDense(32, 10)])

network.build(input_shape=(None, 28*28))
network.summary()

# %%
# 自定义模型
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # 完成网络内需要的网络层的创建工作
        self.fc1 = MyDense(28*28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        # 自定义前向运算逻辑
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x