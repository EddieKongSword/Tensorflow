import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

#create data
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.3+0.6

#create tensorflow structure
Weights=tf.Variable(tf.random_uniform([1], -1,1))
biases=tf.Variable(tf.zeros([1]))
y=x_data*Weights+biases
loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.4)
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(210):
        sess.run(train)
        if i%20==0:
            print(i,sess.run(Weights), sess.run(biases))



