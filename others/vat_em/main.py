import numpy as np
import tensorflow as tf
from vat import VAT
import mnist_model
import os
import BatchDatsetReader as dataset
import matplotlib.pyplot as plt
from six.moves import xrange
import random

os.environ['CUDA_VISIBLE_DEVICES']='1'

batch_size = 32
MAX_ITERATION = int(1e5*1.0 + 1)

x = tf.placeholder(tf.float32, shape=[None, 256, 256, 1], name="input_image")
labels = tf.placeholder(tf.float32, shape=[None, 4,1], name="annotation")
x_ul = tf.placeholder(tf.float32, shape=[None, 256, 256, 1], name="input_image")
keep_prob = tf.placeholder_with_default(1.0, [])

#optimizer = tf.train.GradientDescentOptimizer(0.001)
optimizer = tf.train.AdamOptimizer()

#network = mnist_model.MLPModel(keep_prob=keep_prob)
network = mnist_model.CNNModel(keep_prob=keep_prob)

logits = network(x)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(labels, squeeze_dims=[2]), logits=logits))
vat_cross_entropy, vat_perturbation = VAT(x_ul, network)
predictions = tf.argmax(logits, axis=-1)

train_loss1 = cross_entropy + vat_cross_entropy
train_op1 = optimizer.minimize(train_loss1)

saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

print("Setting up dataset reader")
train_dataset_reader_label = dataset.BatchDatset('D:\\zhangyuhan\\self-supervised\\CellData_OCT\\train\\500')
train_dataset_reader_unlabel = dataset.BatchDatset('E:\\OCT data\\CellData_OCT\\train')
print(len(train_dataset_reader_label.path_list))

print("begining training")
loss_sum_y = []
loss_sum_x = []
plt.scatter(loss_sum_x, loss_sum_y)
plt.ion()
plt.figure(1)

for itr in xrange(MAX_ITERATION):
    
    
    train_images, train_annotations = train_dataset_reader_label.next_batch(batch_size)
    train_images_ul, _ = train_dataset_reader_unlabel.next_batch(batch_size)
    
    feed_dict = {x: train_images, labels: train_annotations, x_ul:train_images_ul, keep_prob: 0.8}
    sess.run(train_op1, feed_dict=feed_dict)

    if itr % 100 == 0:
        train_loss = sess.run(train_loss1, feed_dict=feed_dict)
        print("Step: %d, Train_loss1:%g" % (itr, train_loss))

        loss_sum_y.append(train_loss)
        loss_sum_x.append(itr)
        plt.clf()
        plt.plot(loss_sum_x, loss_sum_y, 'b-', lw=1)
        plt.xlabel('step', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.draw()
        if itr == MAX_ITERATION-1:
            fig = plt.gcf()
            fig.savefig('train_loss.tif', dpi=300)
        plt.pause(0.01)
        

    if itr % 500 == 0:
        saver.save(sess, "logs\model.ckpt", itr)







