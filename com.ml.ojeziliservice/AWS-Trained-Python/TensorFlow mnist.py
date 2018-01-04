
# coding: utf-8

# In[2]:

from tensorflow.examples.tutorials.mnist import input_data


# In[3]:

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[4]:

import tensorflow as tf


# In[5]:

x = tf.placeholder(tf.float32, [None, 784])


# In[6]:

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# In[7]:

y = tf.nn.softmax(tf.matmul(x, W) + b)


# In[8]:

y_ = tf.placeholder(tf.float32, [None, 10])


# In[9]:

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# In[10]:

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[11]:

sess = tf.InteractiveSession()


# In[12]:

tf.global_variables_initializer().run()


# In[13]:

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# In[14]:

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


# In[15]:

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# In[16]:

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# In[17]:

batch = mnist.train.next_batch(100)


# In[18]:

print(mnist.test.labels[0].size)


# In[19]:

print(mnist.test.images[0].size)


# In[20]:

print(sess.run(tf.argmax(y,1), feed_dict={x:mnist.test.images[1:2][:]}))


# In[21]:

mnist.test.images[0:0][:]


# In[25]:

print(sess.run(correct_prediction, feed_dict={x:mnist.test.images[1], y_:mnist.test.labels[1]}))


# In[26]:

y


# In[27]:

print(W)


# In[28]:

batch[0].size


# In[29]:

import numpy as np


# In[30]:

Wv, Bv = sess.run([W, b])


# In[31]:

np.savetxt("Weights.csv", Wv, delimiter=",")


# In[32]:

np.savetxt("BWeights.csv", Bv, delimiter=",")


# In[33]:

signature = tf.saved_model.signature_def_utils.build_signature_def(
    inputs = {'input': tf.saved_model.utils.build_tensor_info(x)},
    outputs = {'output': tf.saved_model.utils.build_tensor_info(y)},
  )


# In[34]:

saved_model_builder = tf.saved_model.builder
mb = saved_model_builder.SavedModelBuilder('/tmp/model')


# In[35]:

mb.add_meta_graph_and_variables(sess,
                                 [tf.saved_model.tag_constants.SERVING],
                                 signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
mb.save() 


# In[81]:

np.savetxt("Seven.txt", mnist.test.images[1:2][:], delimiter=",")


# In[82]:

np.savetxt("Two.txt", mnist.test.images[0:1][:], delimiter=",")


# In[ ]:



