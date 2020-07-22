import tensorflow as tf


def cosine(q,a):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
    pooled_mul_12 = tf.reduce_sum(q * a, 1)
    score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8)
    return tf.reduce_mean(-tf.log(tf.exp(score)))

def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm

def compute_cwise(x0, y0, x_all, y_all, t_len):
    x0_true = tf.where(tf.equal(y,y0),x_all)
    x0_fake = tf.where(tf.logical_not(tf.equal(y,y0)),x_all)
    loss = tf.reduce_mean(tf.nn.l2_loss(tf.nn.softmax(x0)-tf.nn.softmax(x0_true)))/tf.reduce_mean(tf.nn.l2_loss(tf.nn.softmax(x0)-tf.nn.softmax(x0_fake)))
    return loss
    
def cwise_loss(x, xt, y, num_class):
    '''
    x_all = tf.concat([x, xt], -1)
    y_all = tf.concat([y, y], -1)
    x_split = []
    for i in rnage(num_class):
        x_tmp = tf.where(tf.equal(y,i),x)
        x_split.append(x_tmp)
    loss_cwise = 0
    for i in range(num_class):
        r1 = []
        for j in range(num_class):
            if j==i:
                r0 = x_split{j}
            else:
                tf.concat([r1, x_split{j}])
        loss1 = compute(r0, r0)
    '''
    x_all = tf.concat([x, xt], -1)
    y_all = tf.concat([y, y], -1)
    t_len = x_all.get_shape().as_list()[0]
    loss = 0
    for i in range(t_len):
        x0 = x_all[i,:]
        y0 = y_all[i]
        loss_tmp = compute_cwise(x0, y0, x_all, y_all, t_len)
        loss = loss + loss_tmp      
    return tf.reduce_mean(loss)

def loss_label(x, xt, y, xt_r, yt_r, xt_p, yt_p, fcon, fcon_t):
    #f_diff = tf.nn.l2_loss(fcon-fcon_t)
    f_diff = cwise_loss(fcon, fcon_t, y, 4)

    #clipped = lambda t: tf.maximum(t, 1e-30)
    #f_diff = tf.reduce_sum(-tf.reduce_sum(x * tf.log(clipped(xt)), reduction_indices=1) * 1.0) / (tf.reduce_sum(1.0)+1e-30)

    c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.squeeze(y, squeeze_dims=[2])))
    c_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=xt, labels=tf.squeeze(y, squeeze_dims=[2])))

    cr = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=xt_r, labels=yt_r))
    cp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=xt_p, labels=tf.squeeze(yt_p, squeeze_dims=[2])))

    return 1.0*(c+c_t)+0.5*(cr+cp)+0.1*f_diff

def loss_unlabel(x, xt, xt_r, yt_r, xt_p, yt_p, fcon_ul, fcon_t_ul):
    #f_diff = cosine(f,ft)
    f_diff = tf.nn.l2_loss(fcon_ul-fcon_t_ul)

    cr = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=xt_r, labels=yt_r))
    cp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=xt_p, labels=tf.squeeze(yt_p, squeeze_dims=[2])))

    return 0.5*(cr+cp)+0.1*f_diff

def loss_unlabel_em(x, xt):

    #clipped = lambda t: tf.maximum(t, 1e-30)
    
    #f_diff = tf.reduce_sum(-tf.reduce_sum(x * tf.log(clipped(xt)), reduction_indices=1) * 1.0) / (tf.reduce_sum(1.0)+1e-30)

    f_diff = tf.nn.l2_loss(tf.nn.softmax(x)-tf.nn.softmax(xt))

    cr = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=xt_r, labels=yt_r))
    cp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=xt_p, labels=tf.squeeze(yt_p, squeeze_dims=[2])))

    x_label = tf.nn.softmax(x)
    c =  -tf.reduce_mean(tf.reduce_sum(x_label * logsoftmax(x), 1))

    xt_label = tf.nn.softmax(xt)
    c_t =  -tf.reduce_mean(tf.reduce_sum(xt_label * logsoftmax(xt), 1))

    return 0.5*(cr+cp)+0.1*f_diff+1.0*(c+c_t)
    
