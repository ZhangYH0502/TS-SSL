import tensorflow as tf
import tools

def vgg16(x, CLASS_NUM, _dropout, is_training):
    with tf.variable_scope('layer1_1'):
        conv1_1 = tools.conv2d(x, [3,3], 64, 1, is_training, True, True, True)
    with tf.variable_scope('layer1_2'):
        conv1_2 = tools.conv2d(conv1_1, [3,3], 64, 1, is_training, True, True, True)
    with tf.variable_scope('pool1'):
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    with tf.variable_scope('layer2_1'):
        conv2_1 = tools.conv2d(pool1, [3,3], 128, 1, is_training, True, True, True)
    with tf.variable_scope('layer2_2'):
        conv2_2 = tools.conv2d(conv2_1, [3,3], 128, 1, is_training, True, True, True)
    with tf.variable_scope('pool2'):
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    with tf.variable_scope('layer3_1'):
        conv3_1 = tools.conv2d(pool2, [3,3], 256, 1, is_training, True, True, True)
    with tf.variable_scope('layer3_2'):
        conv3_2 = tools.conv2d(conv3_1, [3,3], 256, 1, is_training, True, True, True)
    with tf.variable_scope('layer3_3'):
        conv3_3 = tools.conv2d(conv3_2, [3,3], 256, 1, is_training, True, True, True)
    with tf.variable_scope('pool3'):
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    with tf.variable_scope('layer4_1'):
        conv4_1 = tools.conv2d(pool3, [3,3], 512, 1, is_training, True, True, True)
    with tf.variable_scope('layer4_2'):
        conv4_2 = tools.conv2d(conv4_1, [3,3], 512, 1, is_training, True, True, True)
    with tf.variable_scope('layer4_3'):
        conv4_3 = tools.conv2d(conv4_2, [3,3], 512, 1, is_training, True, True, True)
    with tf.variable_scope('pool4'):
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    with tf.variable_scope('layer5_1'):
        conv5_1 = tools.conv2d(pool4, [3,3], 512, 1, is_training, True, False, True)
    with tf.variable_scope('layer5_2'):
        conv5_2 = tools.conv2d(conv5_1, [3,3], 512, 1, is_training, True, False, True)
    with tf.variable_scope('layer5_3'):
        conv5_3 = tools.conv2d(conv5_2, [3,3], 512, 1, is_training, True, False, True)

    fmp_3 = conv(conv3_3, kernel_size=[1,1], out_channels=256, stride=[1,1,1,1], is_pretrain=_training, bias=False, bn=False, layer_name='conv_3')
    fmp_3 = tf.image.resize_bilinear(fmp_3, [56, 56])
    fmp_4 = conv(conv4_3, kernel_size=[1,1], out_channels=256, stride=[1,1,1,1], is_pretrain=_training, bias=False, bn=False, layer_name='conv_4')
    fmp_4 = tf.image.resize_bilinear(fmp_4, [56, 56])
    fmp_5 = conv(conv5_3, kernel_size=[1,1], out_channels=256, stride=[1,1,1,1], is_pretrain=_training, bias=False, bn=False, layer_name='conv_5')
    fmp_5 = tf.image.resize_bilinear(fmp_5, [56, 56])
    fmp = tf.concat([fmp_3, fmp_4, fmp_5], -1)
    with tf.variable_scope('dilation'):
        fmp_dil_1 = dil_conv(fmp, kernel_size=[3,3], out_channels=256, rate=1, is_pretrain=_training, bias=False, bn=False, layer_name='dilation1')
        fmp_dil_2 = dil_conv(fmp, kernel_size=[3,3], out_channels=256, rate=2, is_pretrain=_training, bias=False, bn=False, layer_name='dilation2')
        fmp_dil_3 = dil_conv(fmp, kernel_size=[3,3], out_channels=256, rate=4, is_pretrain=_training, bias=False, bn=False, layer_name='dilation3')
        fmp_dil_4 = dil_conv(fmp, kernel_size=[3,3], out_channels=256, rate=8, is_pretrain=_training, bias=False, bn=False, layer_name='dilation4')
        fmp_dilation = tf.concat([fmp_dil_1, fmp_dil_2, fmp_dil_3,fmp_dil_4], -1)
        fmp = tools.conv(fmp_dilation, kernel_size=[1,1], out_channels=512, stride=[1,1,1,1], is_pretrain=_training, 
                   bias=False, bn=False,layer_name='conv_dilation')
        
    gap = tf.reduce_mean(fmp, [1, 2])

    with tf.variable_scope('CAM_fc'):
        cam_w = tf.get_variable('CAM_W', shape=[512, CLASS_NUM], initializer=tf.contrib.layers.xavier_initializer(0.0))

    output = tf.matmul(gap, cam_w)

    annotation_pred = tf.argmax(output, axis=-1)
   
    fmp = tf.image.resize_bilinear(fmp, [224, 224])
    
    return annotation_pred, output, fmp

def ResNet(x, _dropout, is_training):
    ResNet_demo = {
        "layer_41": [{"depth": [64, 64, 256], "num_class": 3},
                     {"depth": [128, 128, 512], "num_class": 4},
                     {"depth": [256, 256, 1024], "num_class": 6}],
        
        "layer_50": [{"depth": [64, 64, 256], "num_class": 3},
                     {"depth": [128, 128, 512], "num_class": 4},
                     {"depth": [256, 256, 1024], "num_class": 6},
                     {"depth": [512, 512, 2048], "num_class": 3}],

        "layer_101": [{"depth": [64, 64, 256], "num_class": 3},
                      {"depth": [128, 128, 512], "num_class": 4},
                      {"depth": [256, 256, 1024], "num_class": 23},
                      {"depth": [512, 512, 2048], "num_class": 3}],

        "layer_152": [{"depth": [64, 64, 256], "num_class": 3},
                      {"depth": [128, 128, 512], "num_class": 8},
                      {"depth": [256, 256, 1024], "num_class": 36},
                      {"depth": [512, 512, 2048], "num_class": 3}]
    }
    Res_demo = ResNet_demo["layer_41"]
    layers = []

    # scale1
    with tf.variable_scope('scale1'):
        conv1 = tools.conv2d(x, [7,7], 64, 2, is_training, False, True, True)
    with tf.variable_scope('pool1'):
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")
    layers.append(pool1)

    # scale2,scale3,scale4,scale5
    for k in range(3):
        with tf.variable_scope('scale{}'.format(k+2)):
            for i in range(Res_demo[k]["num_class"]):
                with tf.variable_scope('block{}'.format(i+1)):
                    conv_layer = tools.residual_block(layers[-1], Res_demo[k]["depth"], is_training, first_block=(i == 0), first_stage = (k ==0))
                layers.append(conv_layer)
        
    fc = tf.reduce_mean(layers[-1], [1, 2])

    with tf.variable_scope('CAM_fc'):
        cam_w = tf.get_variable('CAM_W', shape=[fc.get_shape().as_list()[-1], 2],initializer=tf.contrib.layers.xavier_initializer(0.0))
    output = tf.matmul(fc, cam_w)
    
    return output

'''
def concat_dims(logits0, logits1, logits2, logits3, CLASS_NUM):
    dense1 = tf.layers.dense(inputs=tf.concat([logits0, logits1, logits2, logits3], -1),units=1024, activation=tf.nn.relu)
    logits = tf.layers.dense(dense1,units=CLASS_NUM, activation=None)
    return logits
'''

def sample_classify(fc, CLASS_NUM):
    with tf.variable_scope('CAM_fc'):
        cam_w = tf.get_variable('CAM_W', shape=[fc.get_shape().as_list()[-1], CLASS_NUM],initializer=tf.contrib.layers.xavier_initializer(0.0))
    output = tf.matmul(fc, cam_w)
    return output

def pseudo_label_gen(logits):
    logits = tf.nn.softmax(logits,-1)
    pseudo_label = tf.argmax(logits,-1)
    pseudo_label_onehot = tf.one_hot(pseudo_label,4)
    return pseudo_label_onehot

    

