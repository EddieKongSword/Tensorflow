import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
#print("Training data size:", mnist.train.num_examples)

# add layers
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size, out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wxb=tf.matmul(inputs, Weights)+biases
    if activation_function==None:
        outputs=Wxb
    else:
        outputs=activation_function(Wxb)
    return outputs

#compute accuracy
def compute_accuracy(test_images, test_labels):
    global  prediction
    y_pre=sess.run(prediction, feed_dict={xs:test_images})
    correct_number=tf.equal(tf.argmax(y_pre,1), tf.argmax(test_labels,1))  #100*1 bool type
    accuracy=tf.reduce_mean(tf.cast(correct_number, tf.float32))    #bool to float32
    result=sess.run(accuracy, feed_dict={xs:test_images, ys:test_labels})
    return  result

#define placeholder: neural  layer shape
xs=tf.placeholder(dtype=tf.float32, shape=[None,784]) #28*28
ys=tf.placeholder(dtype=tf.float32, shape=[None,10])   #10 outputs

# add output layer
prediction=add_layer(xs, 784, 10, activation_function=tf.nn.sigmoid)    #得到xs*0
#the error between prediction and real data
# cross_entropy=tf.reduce_mean(-tf.reduce_sum(prediction*tf.log(ys),
#                                             reduction_indices=[1]))
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction)
train_step=tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
# print('prediction shape:', prediction.shape)
for i in range(1200):
    batch_xs, batch_ys=mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
    if i%50==0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))


