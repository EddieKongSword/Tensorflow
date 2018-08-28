from tensorflow.examples.tutorials.mnist import input_data
import  tensorflow as tf
# input mnist database
mnist=input_data.read_data_sets("MNIST_data", one_hot=True)

#hyper parameters
learning_rate=0.001
BATCH_SIZE=1000
TIME_STEP=28
INPUT_SIZE=28
train_steps=1000
hidden_num=100
n_class=10

#placeholder
x=tf.placeholder(dtype=tf.float32, shape=[None, 784])
y=tf.placeholder(dtype=tf.float32, shape=[None, 10])
image=tf.reshape(x, shape=[-1, 28, 28])

#define weights
weights=tf.Variable(tf.truncated_normal(shape=[hidden_num,n_class]))

#define biases
biases=tf.Variable(tf.zeros(shape=[n_class]))

#define RNN
# output[:,-1,:],它的形状为[batch_size,rnn_cell.output_size]也就是:
# [batch_size,hidden_num]所以它可以和weights相乘。
# 这就是weights的形状初始化为[hidden_num,n_classes]的原因。
def RNN(x, weights, biases):
    run_cell=tf.nn.rnn_cell.BasicRNNCell(hidden_num)
    outputs, states=tf.nn.dynamic_rnn(run_cell, image, dtype=tf.float32)
    return tf.nn.softmax(tf.matmul(outputs[:, -1, :], weights)+biases,axis=1)

prediction=RNN(x, weights, biases)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
trainer=tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_prediction=tf.equal(tf.argmax(prediction,axis=1),tf.argmax(y, axis=1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
#     labels=tf.argmax(y, axis=1), predictions=tf.argmax(prediction, axis=1),)[1]

#start session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    test_x, test_y = mnist.test.next_batch(BATCH_SIZE)
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        sess.run(trainer, feed_dict={x: batch_x, y: batch_y})
        if i %50==0:
            print(i, sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
