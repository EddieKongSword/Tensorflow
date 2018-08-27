import tensorflow as tf

data1=tf.constant(6)
data2=tf.Variable(2)
add=tf.add(data1, data2)
data_copy=tf.assign(data2, add)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(5):
        print(sess.run(data_copy))

print('end.')