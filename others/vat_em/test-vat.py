from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import scipy
import scipy.misc as misc
import scipy.io

import model1

os.environ['CUDA_VISIBLE_DEVICES']='0'

CLASS_NUM     = 2               #类别数

HEIGHT_IMAGE = 512
WIDTH_IMAGE  = 512
CHANNEL_NUM  = 1


def main(argv=None):
    image = tf.placeholder(tf.float32, shape=[None, HEIGHT_IMAGE, WIDTH_IMAGE, CHANNEL_NUM], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, CLASS_NUM,1], name="annotation")
    keep_prob = tf.placeholder(tf.float32)
    training_flag = tf.placeholder(tf.bool)
    
    with tf.variable_scope('ResNet50'):
        logits  = model1.ResNet(image, keep_prob,training_flag)

    pred = tf.argmax(logits, -1)
  
    
    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    sess = tf.Session()       

    print("global_variables_initializer")
    sess.run(tf.global_variables_initializer())
    
    saver.restore(sess, 'logs1/model.ckpt-50000')
    print("Model restored...")
    

    
    # read test image
    test_images = np.arange(1*HEIGHT_IMAGE*WIDTH_IMAGE*CHANNEL_NUM).reshape(1,HEIGHT_IMAGE,WIDTH_IMAGE,CHANNEL_NUM)
    test_annotations = np.arange(1*CLASS_NUM*1).reshape(1,CLASS_NUM,1)

    filepath = "test"
    filepath_out_results = "semi-supervised_EM_1%"
    dir_or_files = os.listdir(filepath)
    
    for dir_file in dir_or_files:
        results = []
        
        if not os.path.exists(filepath_out_results):
            os.makedirs(filepath_out_results)
            
        print(dir_file)
        image_list = os.listdir(filepath+'\\'+dir_file)
        for filename in image_list:
            print(filename)
    
            imgs = np.array(misc.imread(filepath+'\\'+dir_file+'\\'+filename))
            img_size = np.shape(imgs)
            if len(img_size) < 3:
                test_images[0,:,:,0]=misc.imresize(imgs, [HEIGHT_IMAGE, WIDTH_IMAGE], interp='nearest')
            else:
                test_images[0,:,:,0]=misc.imresize(imgs[:,:,0], [HEIGHT_IMAGE, WIDTH_IMAGE], interp='nearest')
       
            pred1 = sess.run(pred, feed_dict={image: test_images,annotation: test_annotations, keep_prob: 1, training_flag: False})
            results.append(pred1[0])

        scipy.io.savemat(os.path.join(filepath_out_results+"\\"+dir_file+'.mat'), {'results':results})

if __name__ == "__main__":
    tf.app.run()
