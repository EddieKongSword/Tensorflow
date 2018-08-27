from tensorflow.examples.tutorials.mnist import input_data
import  tensorflow as tf
#导入MNIST
mnist=input_data.read_data_sets("MNIST_data", one_hot=True)

#define weights
def weights(shape):
    inital=tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(inital)

#define biases
def biases(shape):
    inital=tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

#define conv2d and pooling layer
# strides=[1, x_movement, y_movement,1]
def conv2d(x,w):
    return tf.nn.conv2d(x,w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#define placeholder for inputs to networks
x=tf.placeholder(dtype=tf.float32, shape=[None, 784])
y=tf.placeholder(dtype=tf.float32, shape=[None, 10])
keep_prob=tf.placeholder(tf.float32)
x_image=tf.reshape(x, shape=[-1,28,28,1])

#convolutional layer 1
#kernel ; shape=[kernel_x, kernel_y, input_depth, output_depth]
w_conv1=weights(shape=[5,5,1,32])
b_conv1=biases([32])       #depth: 32
h_conv1=tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1) #output 28*28*32
h_pool1=max_pool_2x2(h_conv1)                                           #output 14*14*32


#convolutional layer 2
w_conv2=weights(shape=[5,5,32,64])
b_conv2=biases([64])
h_conv2=tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2) #output 14*14*64
h_pool2=max_pool_2x2(h_conv2)                                           #output 7*7*64

# 2D to  1D
h_pool2_shape=h_pool2.get_shape()
h_pool2_dim=int(h_pool2_shape[1]*h_pool2_shape[2]*h_pool2_shape[3])
#fully connect layer 1
w_fc1=weights(shape=[3136,1024])  #3136=7*7*64
b_fc1=biases([1024])
x_fc1=tf.reshape(h_pool2, shape=[-1, 3136])
h_fc1=tf.nn.relu(tf.add(tf.matmul(x_fc1, w_fc1),b_fc1))
fc1_drop_out=tf.nn.dropout(h_fc1, keep_prob=keep_prob)

#fully connect layer 2
w_fc2=weights(shape=[1024,10])
b_fc2=biases([10])
# x_fc1=tf.reshape(h_pool2, shape=[-1, 3136])
prediction=tf.nn.softmax(tf.add(tf.matmul(fc1_drop_out, w_fc2),b_fc2))

loss=tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)
# loss=tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

#prediction accuracy
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_x, batch_y=mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
    if i%50==0:
        print(sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5}))