import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x, y), (x_test, y_test) = mnist.load_data()
x = x / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # 防止 overfitting
    tf.keras.layers.Dense(10, activation='softmax')]
)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

