import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


def add_layer( inputs, in_size, out_size, layer, activation_function=None):
    layer_name=str(layer)
    with tf.name_scope(layer_name):
        Weights=tf.Variable(tf.random_normal([in_size, out_size]),name='W')
        tf.summary.histogram(layer_name+'/weights', Weights)
        biases=tf.Variable(tf.zeros([1])+0.1,name='biases')
        tf.summary.histogram( layer_name+'/biases', biases)
        with tf.name_scope('Wxb'):
            Wxb=tf.matmul(inputs, Weights)+biases
        if activation_function==None:
            outputs=Wxb
        else:
            outputs=activation_function(Wxb)
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

# create data
x_data=np.linspace(-1,1,300)[:, np.newaxis] #[300,] to [300,1], 行向量转列向量
noise=np.random.normal(0,0.04,x_data.shape)
y_data=np.power(x_data,2)+noise
with tf.name_scope('Input'):
    x=tf.placeholder(tf.float32, x_data.shape, name='x_input')
    y=tf.placeholder(tf.float32, y_data.shape,name='y_input')

#create tensorflow layers
L1=add_layer(layer=1, inputs=x, in_size=1, out_size=20, activation_function=tf.nn.relu)
prediction=add_layer(layer=2, inputs=L1, in_size=20,  out_size=1)
with tf.name_scope('Loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(y-prediction),reduction_indices=[1]))
    tf.summary.scalar('loss', loss)   # check scalar
optimizer=tf.train.GradientDescentOptimizer(0.5)
with tf.name_scope('train'):
    train=optimizer.minimize(loss)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    merged=tf.summary.merge_all()  #merge all
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init)
    for i in range(400):
        sess.run(train,feed_dict={x:x_data, y:y_data})
        if i%20==0:
            print(sess.run(loss, feed_dict={x:x_data, y:y_data}))
            results=sess.run(merged, feed_dict={x:x_data, y:y_data})
            writer.add_summary(results, global_step=i)




