from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm
#import numpy as np
import os
from six.moves import xrange
import matplotlib.pyplot as plt
import random

import BatchDatsetReader as dataset
import model1
from vat import VAT

os.environ['CUDA_VISIBLE_DEVICES']='0'

MAX_ITERATION = int(1e5*1.0 + 1)     #最大迭代次数
batch_size    = 4               #batch_size

def main(argv=None):
    image = tf.placeholder(tf.float32, shape=[None, 512, 512, 1], name="input_image")
    annotation = tf.placeholder(tf.float32, shape=[None, 2,1], name="annotation")
    image_ul = tf.placeholder(tf.float32, shape=[None, 512, 512, 1], name="ul_input_image")
    keep_prob = tf.placeholder(tf.float32)
    training_flag = tf.placeholder(tf.bool)

    
    with tf.variable_scope('ResNet50'):
       logits_label_classify   = model1.ResNet(image, keep_prob,training_flag)

    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_label_classify, labels=tf.squeeze(annotation, squeeze_dims=[2])))

    vat_cross_entropy, vat_perturbation = VAT(image_ul, keep_prob, training_flag)

    loss = ce_loss +  vat_cross_entropy

    #训练
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)

    sess = tf.Session()

    print("Setting up Saver...")
    #variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['fc','dim_compress'])
    saver = tf.train.Saver()#variables_to_restore)

    sess = tf.Session()       

    print("global_variables_initializer")
    sess.run(tf.global_variables_initializer())
    
    #saver.restore(sess, 'logs1/model.ckpt-150000')
    #print("Model restored...")

    #读训练样本和验证样本
    print("Setting up dataset reader")
    train_dataset_reader_label = dataset.BatchDatset('train_label_10%')
    train_dataset_reader_unlabel = dataset.BatchDatset('train_unlabel')
    print(len(train_dataset_reader_label.path_list))

    print("begining training")
    
    for itr in xrange(MAX_ITERATION):

        train_images, train_annotations = train_dataset_reader_label.next_batch(batch_size)
        train_images_ul, _ = train_dataset_reader_unlabel.next_batch(batch_size)
        
        feed_dict = {image: train_images, annotation: train_annotations, image_ul:train_images_ul, keep_prob: 0.8, training_flag: True}

        sess.run(train, feed_dict=feed_dict)

        if itr % 100 == 0:
            
            train_loss = sess.run(loss, feed_dict=feed_dict)
            print("Step: %d, Train_loss:%g" % (itr, train_loss))

        if itr % 500 == 0:
            
            saver.save(sess, "logs\model.ckpt", itr)



if __name__ == "__main__":
    tf.app.run()
