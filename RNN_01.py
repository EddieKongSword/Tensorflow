from tensorflow.examples.tutorials.mnist import input_data
import  tensorflow as tf
# input mnist database
mnist=input_data.read_data_sets("MNIST_data", one_hot=True)

#hyper parameters
learning_rate=0.01
BATCH_SIZE=50
TIME_STEP=28
INPUT_SIZE=28


def weights(shape):
    inital=tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(inital)

#define biases
def biases(shape):
    inital=tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

#define RNN
def RNN(x, weights, biases):
    return