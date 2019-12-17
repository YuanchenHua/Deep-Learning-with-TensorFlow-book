import tensorflow as tf

a = tf.constant(1.0)
b = tf.constant(2.0)
c = tf.constant(3.0)
w = tf.constant(5.0)

# 是tf.GrandientTape()
with tf.GradientTape() as tape:
    # tape.watch()
    tape.watch(w)
    y = a*tf.sin(w)+b*w + tf.pow(w,c)

dy_dw = tape.gradient(y,w)
print(dy_dw)

