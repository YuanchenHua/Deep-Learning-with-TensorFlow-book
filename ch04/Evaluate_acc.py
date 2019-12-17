#%%
import  matplotlib
from    matplotlib import pyplot as plt
# Default parameters for plots
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus']=False 
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers
import  os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print(tf.__version__)


def preprocess(x, y): 
    # [b, 28, 28], [b]
    print(x.shape,y.shape)
    # 注意，这里转了浮点型，不转会出错
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)

    return x,y

#%%
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)
#%%
# 处理数据
batchsz = 512
train_db = tf.data.Dataset.from_tensor_slices((x, y))
# shuffle()打乱顺序
train_db = train_db.shuffle(1000)
# 每组512个
train_db = train_db.batch(batchsz)
# 调用预处理 train_db.map(函数) x处理到0～1，y处理成热编码
train_db = train_db.map(preprocess)
# 重复20次，相当于20个epochs
train_db = train_db.repeat(20)

#%%
# 抽一对数据看看
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(batchsz).map(preprocess)
x,y = next(iter(train_db))
print('train sample:', x.shape, y.shape)
print(x[0], y[0])


#%%
def main():

    # learning rate
    lr = 1e-2
    accs,losses = [], []
    # 初始化 w和b
    # 784 => 512
    w1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.zeros([256]))
    # 512 => 256
    w2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1))
    b2 = tf.Variable(tf.zeros([128]))
    # 256 => 10
    w3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))

    # 开始循环计算
    for step, (x,y) in enumerate(train_db):
        # 处理x的shape
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 784))
        with tf.GradientTape() as tape:
            # layer1.
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # layer2
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # output
            out = h2 @ w3 + b3
            # out = tf.nn.relu(out)

            # compute loss
            # [b, 10] - [b, 10]
            loss = tf.square(y-out)
            # [b, 10] => scalar
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3]) 
        for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
            p.assign_sub(lr * g)

        # print
        if step % 80 == 0:
            print(f"step:  {step}  loss:  {float(loss)}")
            losses.append(float(loss))
        
        # 计算准确率
        if step %80 == 0:
            # evaluate/test
            total, total_correct = 0., 0
            # 向前计算
            for x, y in test_db:
                # layer1.
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)
                # layer2
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2 @ w3 + b3
                # [b, 10] => [b]
                # 取出out中最大的那个索引
                pred = tf.argmax(out, axis=1)
                # convert one_hot y to number y
                # 取出y中最大的索引
                y = tf.argmax(y, axis=1)
                # bool type
                # 如果索引一样，即预测正确
                correct = tf.equal(pred, y)
                # bool tensor => int tensor => numpy
                # 先把bool 的正确，转换成int的1，错误转换成0
                # 再把全部的1和0相加， .numpy() 转换成数字，加到之前的total_correct
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                # 计算总个数
                total += x.shape[0]

            print(step, 'Evaluate Acc:', total_correct/total)

            accs.append(total_correct/total)


    plt.figure()
    x = [i*80 for i in range(len(losses))]
    plt.plot(x, losses, color='C0', marker='s', label='训练')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('train.svg')

    plt.figure()
    plt.plot(x, accs, color='C1', marker='s', label='测试')
    plt.ylabel('acc')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('test.svg')

if __name__ == '__main__':
    main()