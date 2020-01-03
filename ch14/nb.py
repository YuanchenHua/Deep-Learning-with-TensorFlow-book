# %%
import tensorflow as tf

model  = tf.keras.Sequential([
    tf.keras.layers.Dense(252,kernel_initializer='he_normal'),
    tf.keras.layers.Dense(64,kernel_initializer='he_normal'),
    tf.keras.layers.Dense(2,kernel_initializer='he_normal')
])
# 0纬
a = tf.constant(3,dtype=tf.float32)
# 1维
a = tf.expand_dims(a,axis=0)
# 2维 进入全连接层,至少2维
a = tf.expand_dims(a,axis=0)
# 去掉没用的第一个纬度
s = model(a)[0]
s
# %%
tf.argmax(s)

# %%
tf.range(5)


# %%
