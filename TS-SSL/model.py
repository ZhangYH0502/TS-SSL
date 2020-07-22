import tensorflow as tf
import tools

def head_mapping(x, CLASS_NUM=4):
    in_dim = x.get_shape().as_list()[-1]
    x = tf.layers.dense(inputs=x, units=in_dim/4, activation=tf.nn.sigmoid, name='m1')
    y = tf.layers.dense(inputs=x, units=in_dim, activation=None, name='m2')
    return tf.nn.softmax(y,-1)

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
        
    fea = tf.reduce_mean(layers[-1], [1, 2])
    '''
    output = tf.layers.dense(inputs=fc, units=4, activation=None, name='class_map')
    output = tf.stop_gradient(output)
    '''
    '''
    in_dim = fc.get_shape().as_list()[-1]
    fc1 = tf.layers.dense(inputs=fc, units=in_dim, activation=tf.nn.sigmoid, name='m1')
    y = tf.layers.dense(inputs=fc1, units=4, activation=None, name='m2')
    '''
    fea_con = head_mapping(fea)
    
    return fea, fea_con

def sample_classify(fc, CLASS_NUM):
    fc = tf.layers.dense(inputs=x, units=CLASS_NUM, activation=None, name='m1')
    return output

def train_model(image, image_t, keep_prob,training_flag):
    with tf.variable_scope('ResNet50'):
        feature_img,   fcon  = ResNet(image, keep_prob,training_flag)
    with tf.variable_scope('ResNet50',reuse=True):
        feature_img_t, fcon_t = ResNet(image_t, keep_prob,training_flag)

    with tf.variable_scope('label_classify'):
        logits_label_classify   = model.sample_classify(feature_img, 2)
    with tf.variable_scope('label_classify',reuse=True):
        logits_label_classify_t = model.sample_classify(feature_img_t, 2)

    with tf.variable_scope('self_rotate'):
        logits_self_rotate = sample_classify(feature_img_t, 16)
        logits_self_rotate = tf.reshape(logits_self_rotate,[-1,4,1,4])

    with tf.variable_scope('self_puzzle'):
        logits_self_puzzle = sample_classify(feature_img_t, 24)
    
    return logits_label_classify, logits_label_classify_t, logits_self_rotate, logits_self_puzzle, fcon, fcon_t

    




