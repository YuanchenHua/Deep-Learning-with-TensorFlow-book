#%%
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

#%%
(x, y), (x_val, y_val) = datasets.mnist.load_data()
# 原来是0到255，先除以255，乘2减1，转换为张量，缩放到0~1
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.-1
y = tf.convert_to_tensor(y, dtype=tf.int32)
# y 本来的shape 为 (60000,)
# y原本的值为0 ～ 9 ，要通过one_hot 编码，因为本身数字大小没意义
y = tf.one_hot(y, depth=10)  # one-hot 编码
print(x.shape, y.shape)

#%%
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(100)

model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)

#%%

def train_epoch(epoch):
    # Step4.loop
    for step, (x, y) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28 * 28))
            # Step1. compute output
            # [b, 784] => [b, 10]
            out = model(x)
            # Step2. compute loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        # Step3. optimize and update w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, model.trainable_variables)
        # w' = w - lr * grad
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())


def train():
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()


# %%
