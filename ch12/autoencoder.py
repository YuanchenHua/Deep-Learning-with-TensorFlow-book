import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import Sequential, layers
from    PIL import Image
from    matplotlib import pyplot as plt



tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def save_images(imgs, path):
    # 新建 Image
    new_im = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            # 在imgs中取一张图片数据
            im = imgs[index]
            # 新建个图
            im = Image.fromarray(im, mode='L')
            # 贴到图框中,坐标(i,j)
            new_im.paste(im, (i, j))
            # 加1,记录
            index += 1
    # 保存图片
    new_im.save(path)

# 隐藏层向量的长度
h_dim = 20
batchsz = 512
lr = 1e-3

# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
# 处理数据
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# we do not need label
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)



class AE(keras.Model):

    def __init__(self):
        # 调用super 的__init__()
        super(AE, self).__init__()

        # Encoders
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])

        # Decoders
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])


    def call(self, inputs, training=None):
        # [b, 784] => [b, 20]
        # 先到隐藏层
        h = self.encoder(inputs)
        # [b, 20] => [b, 784]
        # 恢复为x_hat
        x_hat = self.decoder(h)

        return x_hat



model = AE()
model.build(input_shape=(None, 784))
model.summary()
optimizer = tf.optimizers.Adam(lr=lr)

for epoch in range(100):

    for step, x in enumerate(train_db):

        #[b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            x_rec_logits = model(x)
            # 指定误差
            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)

        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


        if step % 100 ==0:
            print(epoch, step, float(rec_loss))

    # 10个epoch后做一次evaluation
    if (epoch+1) % 10 ==0:
        # 拿到测试集的一组图片x,因为batch size是512,一组就有500张了
        x = next(iter(test_db))
        # 经过model code 和 decode
        logits = model(tf.reshape(x, [-1, 784]))
        # 激活函数一下,得到转换后的 x_hat
        x_hat = tf.sigmoid(logits)

        # 转换后的数据变成图片格式
        # [b, 784] => [b, 28, 28]
        x_hat = tf.reshape(x_hat, [-1, 28, 28])
        x_concat = x_hat
        # 还原回 0~255
        x_concat = x_concat.numpy() * 255.
        # tf格式转换为np格式
        x_concat = x_concat.astype(np.uint8)
        # 调用save_images
        save_images(x_concat, 'rec_epoch_%d.png'%epoch)
