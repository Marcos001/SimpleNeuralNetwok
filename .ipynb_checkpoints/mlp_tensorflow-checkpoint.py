import tensorflow as tf
import numpy as np


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
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    # ModelFnOps connects subgraphs we built to the
    # appropriate functionality.
    return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)

def main():

    #carregando os dados

    from data import  loading_data_train

    matrix = loading_data_train()

    train_X = []
    train_y = []
    test_X = [0, 1, 3]
    test_y = [3, 2, 1]

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

    #treinando a rede

    estimator = tf.contrib.learn.Estimator(model_fn=model)

    # define our data set
    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

    # train -
    estimator.fit(input_fn=input_fn, steps=1000)

    # evaluate our model
    print(estimator.evaluate(input_fn=input_fn, steps=10))




if __name__ == '__main__':
    '''
    '''
    main()
    print('finish run.')