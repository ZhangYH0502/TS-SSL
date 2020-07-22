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
import model
import loss_function1 as lf

os.environ['CUDA_VISIBLE_DEVICES']='0'

MAX_ITERATION = int(1e5*1.0 + 1)     #最大迭代次数
batch_size    = 128               #batch_size

def main(argv=None):
    image = tf.placeholder(tf.float32, shape=[None, 256, 256, 1], name="input_image")
    annotation = tf.placeholder(tf.float32, shape=[None, 4,1], name="annotation")
    image_t = tf.placeholder(tf.float32, shape=[None, 256, 256, 1], name="input_image_t")
    annotation_rotate = tf.placeholder(tf.float32, shape=[None, 4,1,4], name="annotation_rotate")
    annotation_puzzle = tf.placeholder(tf.float32, shape=[None, 24,1], name="annotation_puzzle")

    imageul = tf.placeholder(tf.float32, shape=[None, 256, 256, 1], name="input_image_ul")
    imageul_t = tf.placeholder(tf.float32, shape=[None, 256, 256, 1], name="input_image_t_ul")
    annotation_rotate_ul = tf.placeholder(tf.float32, shape=[None, 4,1,4], name="annotation_rotate_ul")
    annotation_puzzle_ul = tf.placeholder(tf.float32, shape=[None, 24,1], name="annotation_puzzle_ul")

    keep_prob = tf.placeholder(tf.float32)
    training_flag = tf.placeholder(tf.bool)

    w = tf.placeholder(tf.float32)

    logits_x,logits_xt,logits_r, logits_p, fcon, fcon_t = model.train_model(image, image_t, keep_prob,training_flag)
    logits_x_ul,logits_xt_ul,logits_r_ul, logits_p_ul, fcon_ul, fcon_t_ul = model.train_model(imageul, imageul_t, keep_prob,training_flag)

    #loss_label = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_xt, labels=tf.squeeze(annotation, squeeze_dims=[2])))

    loss_label  = lf.loss_label(logits_x, logits_xt, annotation, logits_r, annotation_rotate, logits_p, annotation_puzzle, fcon, fcon_t)
    loss_unlabel = lf.loss_unlabel(logits_xul, logits_xult, logits_r_ul, annotation_rotate_ul, logits_p_ul, annotation_puzzle_ul, fcon_ul, fcon_t_ul)

    loss = loss_label + w*loss_unlabel
    #训练
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)

    sess = tf.Session()

    print("Setting up Saver...")
    #variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['fc','dim_compress'])
    saver = tf.train.Saver()#variables_to_restore)

    sess = tf.Session()       

    print("global_variables_initializer")
    sess.run(tf.global_variables_initializer())
    
    #saver.restore(sess, 'logs-500/model.ckpt-200000')
    #print("Model restored...")

    #读训练样本和验证样本
    print("Setting up dataset reader")
    train_dataset_reader_label = dataset.BatchDatset('D:\\CellData_OCT\\train_label')
    train_dataset_reader_unlabel = dataset.BatchDatset('E:\\CellData_OCT\\train_unlabel')
    #valid_dataset_reader = dataset.BatchDatset('D:\\zhangyuhan\\self-supervised\\CellData_OCT\\eval')
    print(len(train_dataset_reader_label.path_list))
    print(len(train_dataset_reader_unlabel.path_list))

    print("begining training")
    '''
    loss_sum_y = []
    loss_sum_x = []
    plt.scatter(loss_sum_x, loss_sum_y)
    plt.ion()
    plt.figure(1)
    '''
    for itr in xrange(MAX_ITERATION):

        train_images, train_annotations, train_images_t, train_annotations_puzzle,  train_annotations_rotate = train_dataset_reader_label.next_batch(batch_size)
        train_images_ul, train_annotations_ul, train_images_t_ul, train_annotations_puzzle_ul,  train_annotations_rotate_ul = train_dataset_reader_unlabel.next_batch(batch_size)
        
        if itr<5000:
            feed_dict = {image: train_images, annotation: train_annotations,
                     image_t: train_images_t, annotation_puzzle: train_annotations_puzzle, annotation_rotate: train_annotations_rotate,
                     imageul: train_images_ul,
                     imageul_t: train_images_t_ul, annotation_puzzle_ul: train_annotations_puzzle_ul, annotation_rotate_ul: train_annotations_rotate_ul,
                     keep_prob: 0.8, training_flag: True,
                     w: 0.0}
            sess.run(train, feed_dict=feed_dict)
        else if (itr>=0.5e4 and itr<2e4):
            feed_dict = {image: train_images, annotation: train_annotations,
                     image_t: train_images_t, annotation_puzzle: train_annotations_puzzle, annotation_rotate: train_annotations_rotate,
                     imageul: train_images_ul,
                     imageul_t: train_images_t_ul, annotation_puzzle_ul: train_annotations_puzzle_ul, annotation_rotate_ul: train_annotations_rotate_ul,
                     keep_prob: 0.8, training_flag: True,
                     w: (itr-0.5e4)/(2e4-0.5e4)*1.0}
            sess.run(train, feed_dict=feed_dict)
        else:
            feed_dict = {image: train_images, annotation: train_annotations,
                     image_t: train_images_t, annotation_puzzle: train_annotations_puzzle, annotation_rotate: train_annotations_rotate,
                     imageul: train_images_ul,
                     imageul_t: train_images_t_ul, annotation_puzzle_ul: train_annotations_puzzle_ul, annotation_rotate_ul: train_annotations_rotate_ul,
                     keep_prob: 0.8, training_flag: True,
                     w: 1.0}
            sess.run(train, feed_dict=feed_dict)
    
        if itr % 100 == 0:
                
            train_loss = sess.run(loss, feed_dict=feed_dict)
            print("Step: %d, Train_loss:%g" % (itr, train_loss))
            
            '''
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
            '''
        if itr % 500 == 0:
           
            saver.save(sess, "logs\model.ckpt", itr)



if __name__ == "__main__":
    tf.app.run()
