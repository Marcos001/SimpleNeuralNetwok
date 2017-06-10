
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

print('lendo dados > ')
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print('criando variaveis > ')
x = tf.placeholder(tf.float32, [None, 784])