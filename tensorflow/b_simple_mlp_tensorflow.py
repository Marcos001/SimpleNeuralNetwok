# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat


def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)


def main():

    train_X, test_X, train_y, test_y = get_iris_data()

    print('vendo os dados da base > ')
    print('train x = ', len(train_X), ' <> ', train_X)
    print('train y = ', len(train_y), ' <> ',  train_y)

    print('train x = ', len(test_X), ' <> ',  test_X)
    print('test y = ', len(test_y), ' <> ',  test_y)


    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 256                # Number of hidden nodes # 256
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y}))

        #test_accuracy  = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_X, y: test_y}))

        #print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" %(epoch + 1, (100. * train_accuracy), (100. * test_accuracy)))
        print("Epoch = %d, train accuracy = %.2f%%" %(epoch + 1, (100. * train_accuracy)))

    sess.close()


def main_char():
    from data import  loading_data_train

    matrix = loading_data_train()

    train_X = []
    train_y = []
    test_X = [0, 1, 3]
    test_y = [0, 1, 2]

    vetor_x = []

    for i in matrix:
        lis_tmp = []
        try:
            value = float((i[1]))
            train_y.append(int(i[0]))
            lis_tmp.append(value)
            train_X.append(lis_tmp)
            vetor_x.append((train_y, train_X))
        except:
            value = float((i[1]))
            lis_tmp.append(value)
            train_X.append(lis_tmp)
            train_y.append(int(ord(i[0])))



    print('vendo os dados da base > ')

    print('train x = ', len(train_X), ' <> ', train_X)
    print('train y = ', len(train_y), ' <> ',  train_y)

    print('test x = ', len(test_X), ' <> ',  test_X)
    print('test y = ', len(test_y), ' <> ',  test_y)

    def func_quatra():
        import numpy as np
        import tensorflow as tf

        # Declare list of features, we only have one real-valued feature
        def model(features, labels, mode):
            # Build a linear model and predict values
            W = tf.get_variable("W", [1], dtype=tf.float64)
            b = tf.get_variable("b", [1], dtype=tf.float64)
            y = W * features['x'] + b
            # Loss sub-graph
            loss = tf.reduce_sum(tf.square(y - labels))
            # Training sub-graph
            global_step = tf.train.get_global_step()
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            train = tf.group(optimizer.minimize(loss),
                             tf.assign_add(global_step, 1))
            # ModelFnOps connects subgraphs we built to the
            # appropriate functionality.
            return tf.contrib.learn.ModelFnOps(
                mode=mode, predictions=y,
                loss=loss,
                train_op=train)






if __name__ == '__main__':
    main_char()