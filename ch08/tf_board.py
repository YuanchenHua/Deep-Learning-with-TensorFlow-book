import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')]
)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs = 4)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)


summary_writer = tf.summary.create_file_writer('log_dir')
# 整个with部分，应该丢进训练的某个循环里，可以每多少写入一个值
with summary_writer.as_default():
    # 写入测试Loss,不同的名字，只是为了区分数据，可以是loss，可以是acc，可以是任何字符串
    tf.summary.scalar('loss', float(test_loss),step = tf.summary.experimental.set_step(4))
    # 可视化测试用的图片，设置最多可视化 9 张图片
    # tf.summary.image("val-onebyone-images:", x_test,max_outputs=9,tf.summary.experimental.set_step())
