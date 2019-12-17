import tensorflow as tf
import numpy as np

np.random.seed(1)

def generate_data(data_length):
    data = []
    for i in range(data_length):
        x = np.random.uniform(3.,12.)
        eps = np.random.normal(0., 0.01)
        y = 1.477 * x + 0.089 + eps
        data.append([x,y])
    return data

def compute_loss(w,b,data):
    error = 0.
    for i in range(len(data)):
        error += (data[i][0]*w + b - data[i][1])**2
    return error/len(data)

def compute_gradients(w,b,data):
    w_gradient = 0.
    b_gradient = 0.
    for i in range(len(data)):
        x = data[i][0]
        y = data[i][1]
        N = len(data)
        # grad_b = 2(wx+b-y)
        b_gradient += (2/N) * ((w * x + b) - y)
        # grad_w = 2(wx+b-y)*x
        w_gradient += (2/N) * x * ((w * x + b) - y)
    return w_gradient,b_gradient

def gradient_descent(w,b,gradients,learning_rate):
    new_w = w - learning_rate*gradients[0]
    new_b = b - learning_rate*gradients[1]
    return new_w,new_b

def compute_wb(data, starting_b, starting_w, learning_rate, num_iterations):
    w = starting_w
    b = starting_b
    for i in range(num_iterations):
        gradients = compute_gradients(w,b,data)
        w,b = gradient_descent(w,b,gradients,learning_rate)
    return w,b

def run():

    data_length = 1000
    data = generate_data(data_length)
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_w = 0 # initial slope guess
    num_iterations = 10000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_loss(initial_b, initial_w, data))
          )
    print("Running...")
    [w, b] = compute_wb(data, initial_b, initial_w, learning_rate, num_iterations)

    # print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}") 这个格式更快，以后多用
    print("After {0} iterations b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_loss(b, w, data))
          )


if __name__ == '__main__':
    run()





