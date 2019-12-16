import tensorflow as tf 

# 创建4个张量
a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)

# 构建梯度环境
with tf.GradientTape() as tape:
	# 将w加入梯度跟踪列表，watch()是临时加入
	tape.watch([w]) 
	# 构建计算过程
	y = a * w**2 + b * w + c
# 求导，用 tape.gradient(因变量，系数)
[dy_dw] = tape.gradient(y, [w])
print(dy_dw)



