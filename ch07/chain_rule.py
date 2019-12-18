import tensorflow as tf
12121
# 构建待优化变量
x = tf.constant(1.0)
w1 = tf.constant(2.0)
b1 = tf.constant(1.0)
w2 = tf.constant(2.0)
b2 = tf.constant(1.0)
# 要多次调用tape.gradient()需要 persistent=True
with tf.GradientTape(persistent=True) as tape:
    # 非tf.Variable类型的张量需要人为设置记录梯度信息
    # 构建2层网络
    tape.watch([w1,b1,w2,b2])
    y1 = x*tf.sin(w1)+ x**b1
    y2 = y1*tf.sin(w2)+ y1*b2

# 这是gradient返回的结果,是个list 因为会有很多个梯度.这里只有一个,因此0可就是我们要求的这个
# [<tf.Tensor: id=121, shape=(), dtype=float32, numpy=-0.7945481>]
dy2_dw1 = tape.gradient(y2,[w1])[0]
dy2_dy1 = tape.gradient(y2,[y1])[0]
dy1_dw1 = tape.gradient(y1,[w1])[0]

# 验证链式法则
print(dy2_dy1 * dy1_dw1)
print(dy2_dw1)
