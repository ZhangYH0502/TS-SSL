import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm


def batch_normalization(x, is_training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(is_training,
                       lambda : batch_norm(inputs=x, is_training=is_training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=is_training, reuse=True))

def squeeze_excitation_layer(input_x):
    in_dim = input_x.get_shape().as_list()[-1]
    out_dim = in_dim
    squeeze = tf.reduce_mean(input_x, [1, 2])
    excitation = tf.layers.dense(inputs=squeeze, use_bias=False, units=out_dim / 4)
    excitation = tf.nn.relu(excitation)
    excitation = tf.layers.dense(inputs=excitation, use_bias=False, units=out_dim)
    excitation = tf.nn.sigmoid(excitation)
    excitation = tf.reshape(excitation, [-1,1,1,out_dim])
    return input_x * excitation

def conv2d(x, kernel_size, out_channels, stride, is_training, bias=True, bn=True, relu=True):
    in_channels = x.get_shape().as_list()[-1]    
    w = tf.get_variable(name='weights',
                        shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002))
    x = tf.nn.conv2d(x, w, [1,stride,stride,1], padding='SAME', name='conv2d')
    if bias:
        b = tf.get_variable(name='biases',
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, b, name='bias')
    if bn:
        x = batch_normalization(x, is_training, 'bn')
    if relu:
        x = tf.nn.relu(x, name='relu')
    return x

def conv3d(x, kernel_size, out_channels, stride, is_training, bias=True, bn=True, relu=True):
    in_channels = x.get_shape().as_list()[-1]
    w = tf.get_variable(name='weights',
                        shape=[kernel_size[0], kernel_size[1], kernel_size[2], in_channels, out_channels],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002))
    x = tf.nn.conv3d(x, w, [1,1,stride,stride,1], padding='SAME', name='conv3d')
    if bias:
        b = tf.get_variable(name='biases',
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, b, name='bias')
    if bn:
        x = batch_normalization(x, is_training, 'bn')
    if relu:
        x = tf.nn.relu(x, name='relu')
    return x

def dil_conv2d(x, kernel_size, out_channels, rate, is_training, bias=True, bn=True, relu=True):
    in_channels = x.get_shape()[-1]
    w = tf.get_variable(name='weights',
                        shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002))
    x = tf.nn.atrous_conv2d(x, w, rate, padding='SAME', name='dil_conv')
    if bias:
        b = tf.get_variable(name='biases',
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, b, name='bias')
    if bn:
        x = batch_normalization(x, is_training, 'bn')
    if relu:
        x = tf.nn.relu(x, name='relu')
    return x

def residual_block(input_layer, output_channel, is_training=True, first_block=False,first_stage=False):
    f1, f2, f3 = output_channel
    if first_block and not first_stage:
        stride = 2
    else:
        stride = 1
    with tf.variable_scope('a'):
        conv1 = conv2d(input_layer, [1,1], f1, stride, is_training, False, True, True)

    with tf.variable_scope('b'):
        conv2 = conv2d(conv1, [3,3], f2, 1, is_training, False, True, True)

    with tf.variable_scope('c'):
        conv3 = conv2d(conv2, [1,1], f3, 1, is_training, False, True, False)

    with tf.variable_scope('shortcut'):
        if first_block:
            X_shortcut = conv2d(input_layer, [1,1], f3, stride, is_training, False, True, False)
        else:
            X_shortcut = input_layer
    add = tf.add(X_shortcut, conv3)
    output = tf.nn.relu(add)
    return output

