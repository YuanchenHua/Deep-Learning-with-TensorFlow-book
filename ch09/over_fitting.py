# %%
import tensorflow as tf 
from tensorflow.keras import layers, optimizers, datasets, Sequential,regularizers
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import matplotlib.markers
import  numpy as np


N_SAMPLES =1000
N_EPOCHS =500
TRAIN_SIZE = 900

x_min =-2
x_max =3
y_min =-2
y_max =3
OUTPUT_DIR = './ch09/result'

# 从 moon 分布中随机采样 1000 个点
X, y = make_moons(n_samples = N_SAMPLES, noise=0.25, random_state=100)

# 批量作图的方法
def make_plot(X, y, plot_name, file_name, XX=None, YY=None, preds=None):
    plt.figure()
    # sns.set_style("whitegrid")
    axes = plt.gca()
    axes.set_xlim([x_min,x_max])
    axes.set_ylim([y_min,y_max])
    axes.set(xlabel="$x_1$", ylabel="$x_2$")

    # 根据网络输出绘制预测曲面
    if(XX is not None and YY is not None and preds is not None):
        # 把矩阵数据打平,并变成对角线矩阵
        # preds = np.diagflat(preds)
        # 2500个点,重新写成50*50的结构,才能刚好了XX,YY匹配
        preds = preds.reshape(XX.shape)
        plt.contourf(XX, YY, preds, 25, alpha = 0.08,cmap=plt.cm.Spectral)
        plt.contour(XX, YY, preds, levels=[.5],cmap="Greys",vmin=0, vmax=.6)
    # 绘制正负样本
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20,cmap=plt.cm.Spectral)
    # 保存矢量图
    plt.savefig(OUTPUT_DIR+'/'+file_name)

make_plot(X, y, None, "dataset.png")


# 切分数据集
# 下面这个方法,要导入下面这个包
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = TEST_SIZE, random_state=42)
db_X  = tf.convert_to_tensor(X,dtype=tf.float32) 
db_y = tf.convert_to_tensor(y,dtype=tf.int32)

X_train, X_test = tf.split(db_X,(TRAIN_SIZE,N_SAMPLES-TRAIN_SIZE),axis = 0)
y_train, y_test = tf.split(db_y,(TRAIN_SIZE,N_SAMPLES-TRAIN_SIZE),axis = 0)

db_train = tf.data.Dataset.from_tensor_slices((X_train,y_train))
db_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# 得到全部的网格
a = np.linspace(x_min,x_max)
b = np.linspace(y_min,y_max)

XX, YY = np.meshgrid(a,b)
print(XX.shape)

# %% 
# 不同层数
for n in range(5): # 构建 5 种不同层数的网络
    model = Sequential()# 创建容器
    # 创建第一层
    model.add(layers.Dense(8, input_dim=2,activation='relu'))
    for _ in range(n): # 添加 n 层，共 n+2 层
        model.add(layers.Dense(32, activation='relu'))
    # 创建最末层
    model.add(layers.Dense(1, activation='sigmoid')) 
    model.compile(loss='binary_crossentropy', optimizer='adam',
    metrics=['accuracy']) # 模型装配与训练

    history = model.fit(db_train.batch(256), epochs=N_EPOCHS, verbose=1)

    # 绘制不同层数的网络决策边界曲线
    # np.c_()转换成(m,2)的形状
    # XX.ravel()相当于50*50个
    # YY.ravel()相当于50*50个
    # 那么相当于预测2500个点的分类
    preds = model.predict(np.c_[XX.ravel(), YY.ravel()])
    title = "网络层数({})".format(n)
    file_name = f"网络层数{int(n)}.png"
    make_plot(X_train, y_train, title, file_name, XX, YY, preds)

# %%
# 不同dropout层数
for n in range(5): # 构建 5 种不同数量 Dropout 层的网络
    model = Sequential()# 创建
    # 创建第一层
    model.add(layers.Dense(8, input_dim=2,activation='relu'))
    counter = 0

    for _ in range(5): # 网络层数固定为 5
        model.add(layers.Dense(64, activation='relu'))
        if counter < n: # 添加 n 个 Dropout 层
            counter += 1
            model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(1, activation='sigmoid')) # 输出层

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy']) # 模型装配
    # 训练
    history = model.fit(db_train.batch(256), epochs=N_EPOCHS, verbose=1)
    # 绘制不同 Dropout 层数的决策边界曲线
    preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
    title = "Dropout({})".format(n)
    file_name = f"Dropout{n}.png"
    make_plot(X_train, y_train, title, file_name, XX, YY, preds)

# %%%
def build_model_with_regularization(_lambda):
    # 创建带正则化项的神经网络
    model = Sequential()
    # 不带正则化项
    model.add(layers.Dense(8, input_dim=2,activation='relu')) 
    # 带 L2 正则化项
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
    # 带 L2 正则化项
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
    # 带 L2 正则化项
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
    # 输出层
    model.add(layers.Dense(1, activation='sigmoid'))
    # 模型装配
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy']) 
    return model

for _lambda in [1e-5,1e-3,1e-1,0.12,0.13]: # 设置不同的正则化系数
    # 创建带正则化项的模型
    model = build_model_with_regularization(_lambda)
    # 模型训练
    history = model.fit(db_train.batch(256), epochs=N_EPOCHS, verbose=1)
    # 绘制权值范围
    # 绘制不同正则化系数的决策边界线
    preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
    title = f"正则化{_lambda}"
    file = f"正则化{_lambda}.png"
    make_plot(X_train, y_train, title, file, XX, YY, preds)

# %%