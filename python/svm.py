import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 产生双月环数据集
def produceData(r, w, d, num):
    r1 = r - w / 2
    r2 = r + w / 2
    # 上半圆
    theta1 = np.random.uniform(0, np.pi, num)
    X_Col1 = np.random.uniform(r1 * np.cos(theta1), r2 * np.cos(theta1), num)[:, np.newaxis]
    X_Row1 = np.random.uniform(r1 * np.sin(theta1), r2 * np.sin(theta1), num)[:, np.newaxis]
    Y_label1 = np.ones(num)  # 类别标签为1

    # 下半圆
    theta2 = np.random.uniform(-np.pi, 0, num)
    X_Col2 = (np.random.uniform(r1 * np.cos(theta2), r2 * np.cos(theta2), num) + r)[:, np.newaxis]
    X_Row2 = (np.random.uniform(r1 * np.sin(theta2), r2 * np.sin(theta2), num) - d)[:, np.newaxis]
    Y_label2 = -np.ones(num)  # 类别标签为-1,注意：由于采取双曲正切函数作为激活函数，类别标签不能为0

    # 合并
    X_Col1 += np.random.normal(0, 0.55, X_Col1.shape)
    X_Row1 += np.random.normal(0, 0.55, X_Row1.shape)
    X_Col2 += np.random.normal(0, 0.55, X_Col2.shape)
    X_Row2 += np.random.normal(0, 0.55, X_Row2.shape)
    X_Col = np.vstack((X_Col1, X_Col2))
    X_Row = np.vstack((X_Row1, X_Row2))
    X = np.hstack((X_Col, X_Row))
    Y_label = np.hstack((Y_label1, Y_label2))
    Y_label.shape = (num * 2, 1)
    return X, Y_label,X_Col1,X_Row1,X_Col2,X_Row2


def produce_random_data(r, w, d, num):
    X1 = np.random.uniform(-r - w / 2, 2 * r + w / 2, num)
    X2 = np.random.uniform(-r - w / 2 - d, r + w / 2, num)
    X = np.vstack((X1, X2))
    return X.transpose()


def collect_boundary_data(sess, xs, prediction, v_xs):
    # global prediction
    X = np.empty([1, 2])
    X = list()
    for i in range(len(v_xs)):
        x_input = v_xs[i]
        x_input.shape = [1, 2]
        y_pre = sess.run(prediction, feed_dict={xs: x_input})
        if abs(y_pre - 0) < 0.5:
            X.append(v_xs[i])
    return np.array(X)


layers = [2, 20, 1]


def main():
    x_data, y_label,X_Col1,X_Row1,X_Col2,X_Row2 = produceData(10, 6, -6, 1000)
    x_test, y_test,X_Col1_t,X_Row1_t,X_Col2_t,X_Row2_t = produceData(10, 6, -6, 500)
    ###产生空间随机数据
    X_NUM = produce_random_data(10, 6, -6, 5000)

    x = tf.placeholder(tf.float32, [None, 2])
    for i in range(0, len(layers) - 1):
        X = x if i == 0 else y
        node_in = layers[i]
        node_out = layers[i + 1]
        W = tf.Variable(np.random.randn(node_in, node_out).astype('float32') / (np.sqrt(node_in)))
        b = tf.Variable(np.random.randn(node_out).astype('float32'))
        z = tf.matmul(X, W) + b
        # z = tf.contrib.layers.batch_norm(z, center=True, scale=True, is_training=True)
        y = tf.nn.tanh(z)

    y_ = tf.placeholder(tf.float32, [None, 1])
    # Define loss and optimizer
    # 均方误差
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            train_step.run(feed_dict={x: x_data, y_: y_label})
            if i % 1000 == 0:
                train_loss = loss.eval(feed_dict={x: x_data, y_: y_label})
                test_loss = loss.eval(feed_dict={x: x_test, y_: y_test})
                print('step %d, training loss %f%%,testing loss %f%%' % (i, 100*train_loss, 100*test_loss))
        ###边界数据采样
        X_b = collect_boundary_data(sess, x, y, X_NUM)

    ###画出数据
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ###设置坐标轴名称
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.scatter(X_Col1, X_Row1, marker='x', c = 'r')
    ax.scatter(X_Col2, X_Row2, marker='x', c = 'r')
    ax.scatter(X_Col1_t, X_Row1_t, marker='*', c = 'b')
    ax.scatter(X_Col2_t, X_Row2_t, marker='*', c = 'b')
    ###用采样的边界数据拟合边界曲线 7次曲线最佳
    z1 = np.polyfit(X_b[:, 0], X_b[:, 1], 7)
    p1 = np.poly1d(z1)
    x = X_b[:, 0]
    x.sort()
    yvals = p1(x)
    plt.plot(x, yvals, 'y', label='boundray line')
    plt.legend(loc=4)
    plt.show()
    print('DONE!')


if __name__ == '__main__':
    main()
