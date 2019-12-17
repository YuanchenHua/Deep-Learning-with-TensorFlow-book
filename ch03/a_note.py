import tensorflow as tf

# 载入数据
(x_train,y_train),_ = tf.keras.datasets.mnist.load_data()
# 处理数据
batch_size = 200
# 别忘记除到 0～1范围
x_train = x_train/255.0
# 这一步涵盖着 把数据转换成tensor！！！！
# 这个函数是 tf.data.Dataset.from_tensor_slices 且只接受一个参数，所以要括号圈起来
db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db = db.batch(batch_size).repeat(10)

# 建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(128,activation = 'relu'),
    tf.keras.layers.Dense(10)
])
model.build(input_shape = (4,28*28))
optimizer = tf.keras.optimizers.SGD(lr=0.01)
acc_meter = tf.keras.metrics.Accuracy()

# 主程序
for step, (x, y) in enumerate(db):
    with tf.GradientTape() as tape:
        # 处理x，处理y！！！
        x = tf.reshape(x,(-1,28*28))
        y_onehot = tf.one_hot(y,depth=10)
        # 运算结果
        out = model(x)
        # 计算200个的误差
        loss = tf.square(out - y_onehot)
        # 计算每个样本的平均误差，[b]
        loss = tf.reduce_sum(loss) / x.shape[0]

    acc_meter.update_state(tf.argmax(out, axis=1), y)
    # 求梯度  误差，变量
    grads = tape.gradient(loss, model.trainable_variables)
    # 利用优化器更新梯度 zip(梯度，变量)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if step % 10 == 0:
        print(f'step:{step}   loss:{loss}   acc:{acc_meter.result().numpy()}')
        acc_meter.reset_states()


