import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size, out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wxb=tf.matmul(inputs, Weights)+biases
    if activation_function==None:
        outputs=Wxb
    else:
        outputs=activation_function(Wxb)
    return outputs

# create data
x_data=np.linspace(-1,1,300)[:, np.newaxis] #[300,] to [300,1], 行向量转列向量
noise=np.random.normal(0,0.04,x_data.shape)
y_data=np.power(x_data,2)+noise
x=tf.placeholder(tf.float32, x_data.shape)
y=tf.placeholder(tf.float32, y_data.shape)


# # plot data
# plt.scatter(x_data, y_data)
# plt.show()

#create tensorflow layers
L1=add_layer(x, 1, 10, activation_function=tf.nn.relu)
prediction=add_layer(L1, 10, 1)
loss=tf.reduce_mean(tf.reduce_sum(tf.square(y-prediction),reduction_indices=[1]))
# optimizer=tf.train.GradientDescentOptimizer(0.4)
optimizer=tf.train.AdamOptimizer(0.3,)
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    plt.ion()
    for i in range(600):
        sess.run(train,feed_dict={x:x_data, y:y_data})
        if i%20==0:
            # print(sess.run(loss, feed_dict={x:x_data, y:y_data}))
            prediction_values=sess.run(prediction, feed_dict={x:x_data})
            plt.cla()
            plt.scatter(x_data, y_data)
            plt.scatter(x_data, prediction_values,linewidths=0.1)
            plt.pause(0.2)

plt.ioff()
plt.show()

