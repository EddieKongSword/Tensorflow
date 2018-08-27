import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


state=tf.Variable(9,name='counter')
one=tf.constant(1)
new=tf.add(state, one)
update=tf.assign(state, new)
print(state.name)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print(sess.run(state))
