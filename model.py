#coding=utf-8
#https://github.com/WeiTang114/MVCNN-TensorFlow/blob/master/model.py
import tensorflow as tf
import re
import numpy as np
import config
import IPython

DEFAULT_PADDING = 'SAME'
TOWER_NAME = 'tower'
WEIGHT_DECAY_FACTOR = 0. # 3500 -> 2.8

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           initializer=tf.contrib.layers.xavier_initializer())
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _conv(name, in_, ksize, strides=[1,1,1,1], padding=DEFAULT_PADDING, reuse=False):
    
    n_kern = ksize[3]

    with tf.variable_scope(name, reuse=reuse) as scope:
        kernel = _variable_with_weight_decay('weights', shape=ksize, wd=WEIGHT_DECAY_FACTOR)
        conv = tf.nn.conv2d(in_, kernel, strides, padding=padding)
        biases = _variable_on_cpu('biases', [n_kern], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv)

    print name, conv.get_shape().as_list()
    return conv

def _maxpool(name, in_, ksize, strides, padding=DEFAULT_PADDING):
    pool = tf.nn.max_pool(in_, ksize=ksize, strides=strides,
                          padding=padding, name=name)

    print name, pool.get_shape().as_list()
    return pool

def _fc(name, in_, outsize, dropout=1.0, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        
        insize = in_.get_shape().as_list()[-1]
        weights = _variable_with_weight_decay('weights', shape=[insize, outsize], wd=0.004)
        biases = _variable_on_cpu('biases', [outsize], tf.constant_initializer(0.0))
        fc = tf.nn.relu(tf.matmul(in_, weights) + biases, name=scope.name)
        fc = tf.nn.dropout(fc, dropout)

        _activation_summary(fc)

    

    print name, fc.get_shape().as_list()
    return fc

def inference(img, keep_prob, feature, reuse=True):


    conv1 = _conv('conv1', img, [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', reuse=reuse)
    pool1 = _maxpool('pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv2 = _conv('conv2', pool1, [5, 5, 96, 256], reuse=reuse)
    pool2 = _maxpool('pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    conv3 = _conv('conv3', pool2, [3, 3, 256, 384], reuse=reuse)

    conv4 = _conv('conv4', conv3, [3, 3, 384, 384], reuse=reuse)

    conv5 = _conv('conv5', conv4, [3, 3, 384, 256], reuse=reuse)
    pool5 = _maxpool('pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    shape = pool5.get_shape().as_list()  
    pool5_vector = tf.reshape(pool5, [-1, np.prod(shape[1:])])

    if feature == "pool5":
        return pool5_vector

    fc6 = _fc('fc6', pool5_vector, 4096, dropout=keep_prob, reuse= reuse)
    
    if feature == "fc6":
        return fc6

    fc7 = _fc('fc7', fc6, 4096, dropout=keep_prob, reuse= reuse)
 
    if feature == "fc7":
        return fc7   
    #fc8 = _fc('fc8', fc7, n_classes)

def inference_crossview_not_share(views, keep_prob, feature, origin_reuse=True):
 
    views_pool = []
    reuse = origin_reuse
    for i in xrange(len(views)): 
        if not origin_reuse:
            reuse = (i != 0)
        if i == 0: #search
            reuse = True
            conv1 = _conv('conv1', views[i], [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', reuse=reuse)
            pool1 = _maxpool('pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
 
            conv2 = _conv('conv2', pool1, [5, 5, 96, 256], reuse=reuse)
            pool2 = _maxpool('pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
 
            conv3 = _conv('conv3', pool2, [3, 3, 256, 384], reuse=reuse)
 
            conv4 = _conv('conv4', conv3, [3, 3, 384, 384], reuse=reuse)
 
            conv5 = _conv('conv5', conv4, [3, 3, 384, 256], reuse=reuse)
            pool5 = _maxpool('pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        elif i== 1: #street-view
            reuse = False
            conv1 = _conv('cv_conv1', views[i], [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', reuse=reuse)
            pool1 = _maxpool('cv_pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
 
            conv2 = _conv('cv_conv2', pool1, [5, 5, 96, 256], reuse=reuse)
            pool2 = _maxpool('cv_pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
 
            conv3 = _conv('cv_conv3', pool2, [3, 3, 256, 384], reuse=reuse)
 
            conv4 = _conv('cv_conv4', conv3, [3, 3, 384, 384], reuse=reuse)
 
            conv5 = _conv('cv_conv5', conv4, [3, 3, 384, 256], reuse=reuse)
            pool5 = _maxpool('cv_pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')           
        elif i== 2: #aerial
            reuse = False
            conv1 = _conv('aerial_conv1', views[i], [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', reuse=reuse)
            pool1 = _maxpool('aerial_pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
 
            conv2 = _conv('aerial_conv2', pool1, [5, 5, 96, 256], reuse=reuse)
            pool2 = _maxpool('aerial_pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
 
            conv3 = _conv('aerial_conv3', pool2, [3, 3, 256, 384], reuse=reuse)
 
            conv4 = _conv('aerial_conv4', conv3, [3, 3, 384, 384], reuse=reuse)
 
            conv5 = _conv('aerial_conv5', conv4, [3, 3, 384, 256], reuse=reuse)
            pool5 = _maxpool('aerial_pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID') 
 
 
        shape = pool5.get_shape().as_list()  
        pool5_vector = tf.reshape(pool5, [-1, np.prod(shape[1:])])
        views_pool.append(pool5_vector)
 
    pool5_vp = _view_pool(views_pool, 'cv_pool5_vp')
 
 
    if feature == "pool5":
        return pool5_vp
 
 
 
    fc6 = _fc('fc6', pool5_vp, 4096, dropout=keep_prob, reuse=origin_reuse)
     
    if feature == "fc6":
        return fc6
 
    fc7 = _fc('fc7', fc6, 4096, dropout=keep_prob, reuse=origin_reuse)
  
    if feature == "fc7":
        return fc7   
    #fc8 = _fc('fc8', fc7, n_classes)

def inference_crossview_pool5(views, keep_prob, feature, origin_reuse=True):
 
    views_pool = []
    reuse = origin_reuse
    for i in xrange(len(views)): 
        if not origin_reuse:
            reuse = (i != 0)
        conv1 = _conv('conv1', views[i], [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', reuse=reuse)
        pool1 = _maxpool('pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
 
        conv2 = _conv('conv2', pool1, [5, 5, 96, 256], reuse=reuse)
        pool2 = _maxpool('pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
         
        conv3 = _conv('conv3', pool2, [3, 3, 256, 384], reuse=reuse)
 
        conv4 = _conv('conv4', conv3, [3, 3, 384, 384], reuse=reuse)
 
        conv5 = _conv('conv5', conv4, [3, 3, 384, 256], reuse=reuse)
        pool5 = _maxpool('pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
 
         
        shape = pool5.get_shape().as_list()  
        pool5_vector = tf.reshape(pool5, [-1, np.prod(shape[1:])])
 
 
        views_pool.append(pool5_vector)
 
    pool5_vp = _view_pool(views_pool, 'pool5_vp')
 
 
    if feature == "pool5":
        return pool5_vp
 
 
 
    fc6 = _fc('fc6', pool5_vp, 4096, dropout=keep_prob, reuse = origin_reuse)
     
    if feature == "fc6":
        return fc6
 
    fc7 = _fc('fc7', fc6, 4096, dropout=keep_prob, reuse=origin_reuse)
  
    if feature == "fc7":
        return fc7   
    #fc8 = _fc('fc8', fc7, n_classes)

def inference_crossview_fc6_max(views, keep_prob, feature, origin_reuse=True):
 
    views_pool = []
    reuse = origin_reuse
    for i in xrange(len(views)): 
        if not origin_reuse:
            reuse = (i != 0)
        conv1 = _conv('conv1', views[i], [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', reuse=reuse)
        pool1 = _maxpool('pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
 
        conv2 = _conv('conv2', pool1, [5, 5, 96, 256], reuse=reuse)
        pool2 = _maxpool('pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
         
        conv3 = _conv('conv3', pool2, [3, 3, 256, 384], reuse=reuse)
 
        conv4 = _conv('conv4', conv3, [3, 3, 384, 384], reuse=reuse)
 
        conv5 = _conv('conv5', conv4, [3, 3, 384, 256], reuse=reuse)
        pool5 = _maxpool('pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
 
         
        shape = pool5.get_shape().as_list()  
        pool5_vector = tf.reshape(pool5, [-1, np.prod(shape[1:])])
 

        if feature == "pool5":
            return pool5_vector



        fc6 = _fc('fc6', pool5_vector, 4096, dropout=keep_prob, reuse = reuse)
 
        views_pool.append(fc6)
 
    fc6_vp = _view_pool(views_pool, 'fc6_vp')
 
 

     
    if feature == "fc6":
        return fc6_vp
 
    fc7 = _fc('fc7', fc6, 4096, dropout=keep_prob, reuse=origin_reuse)
  
    if feature == "fc7":
        return fc7   

def inference_crossview(views, keep_prob, feature, origin_reuse=True):
 
    views_pool = []
    for i in xrange(len(views)): 
        if not origin_reuse:
            reuse = (i != 0)
        conv1 = _conv('conv1', views[i], [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', reuse=reuse)
        pool1 = _maxpool('pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
 
        conv2 = _conv('conv2', pool1, [5, 5, 96, 256], reuse=reuse)
        pool2 = _maxpool('pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
         
        conv3 = _conv('conv3', pool2, [3, 3, 256, 384], reuse=reuse)
 
        conv4 = _conv('conv4', conv3, [3, 3, 384, 384], reuse=reuse)
 
        conv5 = _conv('conv5', conv4, [3, 3, 384, 256], reuse=reuse)
        pool5 = _maxpool('pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
 
         
        shape = pool5.get_shape().as_list()  
        pool5_vector = tf.reshape(pool5, [-1, np.prod(shape[1:])])
        views_pool.append(pool5_vector)



    two_pool5 = cv_mean(views_pool, 'cv_mean')
    views_pool = []
    for i in xrange(len(two_pool5)): 
        if not origin_reuse:
            reuse = (i != 0)
        fc6 = _fc('fc6', two_pool5[i][0], 4096, dropout=keep_prob, reuse = reuse)
        views_pool.append(fc6)
 

    fc6_vp = _view_pool(views_pool, 'fc6_vp')
 
 

     
    if feature == "fc6":
        return fc6_vp
 
    fc7 = _fc('fc7', fc6, 4096, dropout=keep_prob, reuse=origin_reuse)
  
    if feature == "fc7":
        return fc7   

def inference_crossview_3cvmean(views, keep_prob, feature, origin_reuse=True):
 
    views_pool = []
    for i in xrange(len(views)): 
        if not origin_reuse:
            reuse = (i != 0)
        conv1 = _conv('conv1', views[i], [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', reuse=reuse)
        pool1 = _maxpool('pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
 
        conv2 = _conv('conv2', pool1, [5, 5, 96, 256], reuse=reuse)
        pool2 = _maxpool('pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
         
        conv3 = _conv('conv3', pool2, [3, 3, 256, 384], reuse=reuse)
 
        conv4 = _conv('conv4', conv3, [3, 3, 384, 384], reuse=reuse)
 
        conv5 = _conv('conv5', conv4, [3, 3, 384, 256], reuse=reuse)
        pool5 = _maxpool('pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
 
         
        shape = pool5.get_shape().as_list()  
        pool5_vector = tf.reshape(pool5, [-1, np.prod(shape[1:])])
        views_pool.append(pool5_vector)



    two_pool5 = cv_mean_3(views_pool, 'cv_mean')
    views_pool = []
    for i in xrange(len(two_pool5)): 
        if not origin_reuse:
            reuse = (i != 0)
        fc6 = _fc('fc6', two_pool5[i][0], 4096, dropout=keep_prob, reuse = reuse)
        views_pool.append(fc6)
 

    fc6_vp = _view_pool(views_pool, 'fc6_vp')
 
 

     
    if feature == "fc6":
        return fc6_vp
 
    fc7 = _fc('fc7', fc6, 4096, dropout=keep_prob, reuse=origin_reuse)
  
    if feature == "fc7":
        return fc7  

def _view_pool(view_features, name):
    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]
    for v in view_features[1:]:

        v = tf.expand_dims(v, 0)
        vp = tf.concat(0, [vp, v]) #old version

    print 'vp before reducing:', vp.get_shape().as_list()
    vp = tf.reduce_max(vp, [0], name=name)
    print 'vp after reducing:', vp.get_shape().as_list()
    return vp

def cv_mean_3(view_features, name):
    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]

    print(vp.get_shape().as_list)
    half = vp.get_shape().as_list()[2]/2
    zeros = tf.zeros_like(vp)

    vp_down = tf.concat(2, [vp[:, :, :half], zeros[:,:,half:]])
    vp_up = tf.concat(2, [zeros[:,:,:half], vp[:,:,half:]])

    # street-view
    v = view_features[1]
    v = tf.expand_dims(v, 0)
    vp_up = tf.concat(0, [vp_up, v])
    vp_up = tf.reduce_mean(vp_up, [0], name=name+"_up")
    vp_up = tf.expand_dims(vp_up, 0)



    # aerial
    v = view_features[2]
    v = tf.expand_dims(v, 0)
    vp_down = tf.concat(0, [vp_down, v])
    vp_down = tf.reduce_mean(vp_down, [0], name=name+"_down")
    vp_down = tf.expand_dims(vp_down, 0)

    #merge

    three_pool5 = [vp_up, vp, vp_down]
    #print 'vp_pool5 after mean:', two_pool5.get_shape().as_list()
    return three_pool5

def cv_mean(view_features, name):
    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]

    print(vp.get_shape().as_list)
    half = vp.get_shape().as_list()[2]/2
    zeros = tf.zeros_like(vp)

    vp_down = tf.concat(2, [vp[:, :, :half], zeros[:,:,half:]])
    vp_up = tf.concat(2, [zeros[:,:,:half], vp[:,:,half:]])

    # street-view
    v = view_features[1]
    v = tf.expand_dims(v, 0)
    vp_up = tf.concat(0, [vp_up, v])
    vp_up = tf.reduce_mean(vp_up, [0], name=name+"_up")
    vp_up = tf.expand_dims(vp_up, 0)



    # aerial
    v = view_features[2]
    v = tf.expand_dims(v, 0)
    vp_down = tf.concat(0, [vp_down, v])
    vp_down = tf.reduce_mean(vp_down, [0], name=name+"_down")
    vp_down = tf.expand_dims(vp_down, 0)

    #merge

    two_pool5 = [vp_up, vp_down]
    #print 'vp_pool5 after mean:', two_pool5.get_shape().as_list()
    return two_pool5

def not_padding_zero_cv_mean(view_features, name):
    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]

    print(vp.get_shape().as_list)
    half = vp.get_shape().as_list()[2]/2
    zeros = tf.zeros_like(vp)

    
    # street-view
    v = view_features[1]
    v = tf.expand_dims(v, 0)
    vp_up = tf.concat(2, [vp[:,:,half:], v[:,:,:half]])

    vp_up = tf.concat(0, [vp_up, v])
    vp_up = tf.reduce_mean(vp_up, [0], name=name+"_up")
    vp_up = tf.expand_dims(vp_up, 0)


    # aerial
    v = view_features[2]
    v = tf.expand_dims(v, 0)
    vp_down = tf.concat(2, [v[:,:,half:], vp[:, :, :half]])
    vp_down = tf.concat(0, [vp_down, v])

    vp_down = tf.reduce_mean(vp_down, [0], name=name+"_down")
    vp_down = tf.expand_dims(vp_down, 0)

    #merge

    two_pool5 = [vp_up, vp_down]
    #print 'vp_pool5 after mean:', two_pool5.get_shape().as_list()
    return two_pool5

def _ggview_pool(view_features, name):

    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]

    print(vp.get_shape().as_list)
    half = vp.get_shape().as_list()[2]/2
    zeros = tf.zeros_like(vp)

    vp_down = tf.concat(2, [vp[:, :, :half], zeros[:,:,half:]])
    vp_up = tf.concat(2, [zeros[:,:,:half], vp[:,:,half:]])

    # street-view
    v = view_features[1]
    v = tf.expand_dims(v, 0)
    vp_up = tf.concat(0, [vp_up, v])
    vp_up = tf.reduce_mean(vp_up, [0], name=name+"_up")
    vp_up = tf.expand_dims(vp_up, 0)



    # aerial
    v = view_features[2]
    v = tf.expand_dims(v, 0)
    vp_down = tf.concat(0, [vp_down, v])
    vp_down = tf.reduce_mean(vp_down, [0], name=name+"_down")
    vp_down = tf.expand_dims(vp_down, 0)

    #merge

    vp = tf.concat(0, [vp_up, vp_down])

    print 'vp before reducing:', vp.get_shape().as_list()
    vp = tf.reduce_max(vp, [0], name=name)
    print 'vp after reducing:', vp.get_shape().as_list()
    return vp

def eval_loss(pair1, pair2):    
    dis = tf.reduce_sum(tf.square(pair1 - pair2), 1)
    return tf.reduce_mean(dis)


def triplet_loss(anchor, positive, negative):
    margin = 0.5

    d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
    d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)

    #loss_origin = tf.maximum(0., margin + tf.sqrt(d_pos) - tf.sqrt(d_neg))
    loss_origin = tf.maximum(0., margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss_origin)
    return loss, d_pos, d_neg, loss_origin

def load_alexnet(sess, caffetf_modelpath, layer_name=''):
    """ caffemodel: np.array, """

    def load(name, layer_data, group=1):
        w, b = layer_data

        if group != 1:
            w = np.concatenate((w, w), axis=2) 

        with tf.variable_scope(name, reuse=True):
            for subkey, data in zip(('weights', 'biases'), (w, b)):
                print 'loading ', name, subkey
                var = tf.get_variable(subkey)
                sess.run(var.assign(data))

    caffemodel = np.load(caffetf_modelpath)
    data_dict = caffemodel.item()
    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']:
        name = layer_name + l

        # historical grouping by alexnet
        if l == 'conv2' or l == 'conv4' or l == 'conv5':
            load(name, data_dict[l], group=2)
        else:
            try:
                load(name, data_dict[l])
            except:
                print('not load {}'.format(l))

def load_alexnet_place(sess, caffetf_modelpath, layer_name=''):
    """ caffemodel: np.array, """
    
    def load(name, layer_data, group=1):
        w = layer_data['weights']
        b = layer_data['biases']
        if group != 1:
            w = np.concatenate((w, w), axis=2) 

        with tf.variable_scope(name, reuse=True):
            for subkey, data in zip(('weights', 'biases'), (w, b)):
                print 'loading ', name, subkey
                var = tf.get_variable(subkey)
                sess.run(var.assign(data))

    caffemodel = np.load(caffetf_modelpath)
    data_dict = caffemodel.item()
    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']:
        name = layer_name + l

        # historical grouping by alexnet
        if l == 'conv2' or l == 'conv4' or l == 'conv5':
            load(name, data_dict[l], group=2)
        else:
            try:
                load(name, data_dict[l])
            except:
                print('not load {}'.format(l))

def train(total_loss, global_step, data_size):
    num_batches_per_epoch = data_size / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)
    
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    
    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad,var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def feature_normalize(feature_list):
    feature_norm = []
    for feature in feature_list:
        feature_norm.append(tf.nn.l2_normalize(feature, dim=1, name='normalized'))
    return feature_norm