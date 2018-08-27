import tensorflow as tf

a = tf.Variable(tf.random_normal([2,2]))
b = tf.Variable(tf.truncated_normal([2,2],seed=2))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print('\n')
    print(sess.run(b))
