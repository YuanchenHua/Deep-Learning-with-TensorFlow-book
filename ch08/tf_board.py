import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,Sequential,losses,optimizers,datasets,metrics

''' 
新建测量器，写入数据，读取统 计数据和清零测量器。

# 新建平均测量器，适合 Loss 数据
loss_meter = metrics.Mean()

# 记录采样的数据,上述采样代码放置在每个 batch 运算完成后，测量器会自动根据采样的数据来统计平均值。
loss_meter.update_state(float(loss))

# 在采样多次后，可以测量器的 result()函数获取统计值
print(step, 'loss:', loss_meter.result())

由于测量器会统计所有历史记录的数据，因此在合适的时候有必要清除历史状态。
通过 reset_states()即可实现。例如，在每次读取完平均误差后，清零统计信息，以便下一轮统计的开始：
if step % 100 == 0:
    print(step, 'loss:', loss_meter.result())
    loss_meter.reset_states() 
'''

# 载入数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 处理数据
x_train = tf.convert_to_tensor(x_train/255.)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_train = tf.one_hot(y_train,depth=10)

# 
db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(64)


fc1 = tf.keras.layers.Dense(256, activation='relu')
fc2 = tf.keras.layers.Dense(128, activation='relu')
fc3 = tf.keras.layers.Dense(10, activation='softmax')
model = tf.keras.Sequential([fc1, fc2, fc3])
optimizer = tf.keras.optimizers.Adam(lr = 0.01)
# 写log
summary_writer = tf.summary.create_file_writer('log_dir')
# 新建测量器
loss_meter = metrics.Mean()

# 主循环
for i in range(20):
    losses = 0.0
    for step, (x, y) in enumerate(db_train):
        with tf.GradientTape() as tape:
            x = tf.reshape(x,(-1,28*28))
            out = model(x)
            loss = tf.keras.losses.categorical_crossentropy(y, out)
            # lose.shape 是(64,) 因此求reduce_sum 然后除以形状,用reduce_mean同效果
            loss = tf.reduce_sum(loss)/x.shape[0]
    
        # grads = tape.gradient(loss, model.trainable_variables)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # losses +=loss
        # 记录采样的数据,上述采样代码放置在每个 batch 运算完成后，测量器会自动根据采样的数据来统计平均值。
        loss_meter.update_state(loss)
       
    with summary_writer.as_default():
        # 写入测试Loss,不同的名字，只是为了区分数据，可以是loss，可以是acc，可以是任何字符串
        tf.summary.scalar('loss', float(loss_meter.result()),step = i)
    print(loss_meter.result())
    loss_meter.reset_states()




# 整个with部分，应该丢进训练的某个循环里，可以每多少写入一个值
# with summary_writer.as_default():
    # 写入测试Loss,不同的名字，只是为了区分数据，可以是loss，可以是acc，可以是任何字符串
    # tf.summary.scalar('loss', float(test_loss),step = tf.summary.experimental.set_step(4))
    # 可视化测试用的图片，设置最多可视化 9 张图片
    # tf.summary.image("val-onebyone-images:", x_test,max_outputs=9,tf.summary.experimental.set_step())
