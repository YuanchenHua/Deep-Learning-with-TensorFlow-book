import tensorflow as tf
import numpy as np
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

# 主循环
for i in range(20):
    losses = 0.0
    for step, (x, y) in enumerate(db_train):
        with tf.GradientTape() as tape:
            x = tf.reshape(x,(-1,28*28))
            out = model(x)
            loss = tf.keras.losses.categorical_crossentropy(y, out)
            loss = tf.reduce_sum(loss)/x.shape[0]
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses +=loss
    

    print(losses)
    with summary_writer.as_default():
        # 写入测试Loss,不同的名字，只是为了区分数据，可以是loss，可以是acc，可以是任何字符串
        tf.summary.scalar('loss', float(losses),step = i)



# 整个with部分，应该丢进训练的某个循环里，可以每多少写入一个值
# with summary_writer.as_default():
    # 写入测试Loss,不同的名字，只是为了区分数据，可以是loss，可以是acc，可以是任何字符串
    # tf.summary.scalar('loss', float(test_loss),step = tf.summary.experimental.set_step(4))
    # 可视化测试用的图片，设置最多可视化 9 张图片
    # tf.summary.image("val-onebyone-images:", x_test,max_outputs=9,tf.summary.experimental.set_step())
