import tensorflow as tf
import numpy as np
# 载入数据
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
# 建立tf.Variable
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros(256))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros(128))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros(10))

# 一定要转浮点，不知道为毛！！！
# 一定要除以255，归结到0-1范围，不然计算误差是，会出现无限接近0，从而没法优化
x_train = tf.cast(x_train,tf.float32)/255
# 为了方便，只拿1000个数据训练把
x_train = x_train[0:1000,...]
y_train = y_train[0:1000,...]
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_dataset = train_dataset.batch(100)

lr = 1e-3
def train_epoch(epoch, lr):
    # Step4.loop
    for x, y in zip(x_train,y_train):
        x = tf.reshape(x, (-1, 28 * 28))
        with tf.GradientTape() as tape:


            h1 = tf.matmul(x,w1) + tf.broadcast_to(b1, [x.shape[0], 256])
            h1 = tf.nn.relu(h1)

            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)

            h3 = h2 @ w3 + b3
            # h3 = tf.nn.relu(h3)

            y = tf.one_hot(y, depth=10)

            loss = tf.reduce_mean(tf.square(h3 - y))

        # Step3. optimize and update w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

    print(epoch, 'loss:', loss.numpy())


def train():
    for epoch in range(30):
        train_epoch(epoch, lr)


if __name__ == '__main__':
    train()
