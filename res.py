# -*- coding: utf-8 -*-

import tensorflow as tf

# definitions for ResNet


res_conv1_fmaps = 64
res_conv1_ksize = 3
res_conv1_stride = 1
res_conv1_pad = 'SAME'

res_pool2_ksize = [1,3,3,1]
res_pool2_stride = [1,2,2,1]
res_pool2_pad = 'SAME'


# mod
res_mod_stride = 2
res_mod_pad = 'SAME'
res_mod_ksize = 3


res_pool3_kisze = [1,4,4,1]
res_pool3_stride = [1,1,1,1]
res_pool3_pad = 'VALID'

res_nfc = 1000

res_activation = tf.nn.relu

res_bn_momentum = 0.9
#################################
#  end of ResNet definitions
#################################




    
def build_res(X_input, nout, training):
    def res_mod(pre, fmaps, ksize, stride,pad, train = training):
    
    
        p1 = tf.layers.conv2d(pre, filters = fmaps,
                              kernel_size = 1,
                              strides = stride,
                              padding = pad)
        
        p1_bn = tf.layers.batch_normalization(p1, momentum=res_bn_momentum,
                                              training = train)
        
        p2_1 = tf.layers.conv2d(pre, filters = fmaps,
                                kernel_size = ksize,
                                strides = stride,
                                padding = pad)
        
        p2_1_bn = tf.layers.batch_normalization(p2_1, momentum=res_bn_momentum,
                                                training = train)
        
        p2_1_bn_ac = res_activation(p2_1_bn)
        
        p2_2 = tf.layers.conv2d(p2_1_bn_ac, filters = fmaps,
                                kernel_size = ksize,
                                strides = 1,
                                padding = pad)
        
        p2_2_bn = tf.layers.batch_normalization(p2_2, momentum=res_bn_momentum,
                                                training = train)
        
        
        result = tf.concat((p1_bn, p2_2_bn), axis = 3)
        return res_activation(result)
    
    with tf.name_scope('ResNet'):
        res_conv1 = tf.layers.conv2d(X_input, filters = res_conv1_fmaps,
                                     kernel_size = res_conv1_ksize,
                                     strides = res_conv1_stride,
                                     padding = res_conv1_pad,
                                     activation = res_activation,
                                     name = 'res_conv1')
        
        res_pool2 = tf.nn.max_pool(res_conv1, ksize = res_pool2_ksize,
                                   strides = res_pool2_stride,
                                   padding = res_pool2_pad,
                                   name = 'res_pool2')
        
        res_mod1 = res_mod(res_pool2, 64, 3, 1, 'SAME', training)
        res_mod2 = res_mod(res_mod1, 64, 3, 1, 'SAME', training)
        res_mod3 = res_mod(res_mod2, 128, 3, 2, 'SAME', training)
        res_mod4 = res_mod(res_mod3, 128, 3, 1, 'SAME', training)
        res_mod5 = res_mod(res_mod4, 256, 3, 2, 'SAME', training)
        res_mod6 = res_mod(res_mod5, 256, 3, 1, 'SAME', training)
        res_mod7 = res_mod(res_mod6, 512, 3, 2, 'SAME', training)
        res_mod8 = res_mod(res_mod7, 512, 3, 1, 'SAME', training)
        res_mod7 = res_mod(res_mod6, 1024, 3, 2, 'SAME', training)
        res_mod8 = res_mod(res_mod7, 1024, 3, 1, 'SAME', training)
        
        res_pool3 = tf.nn.avg_pool(res_mod8, ksize = res_pool3_kisze,
                                   strides = res_pool3_stride,
                                   padding = res_pool3_pad,
                                   name = 'res_pool3')
        
        res_pool3_flat = tf.reshape(res_pool3, shape = [-1,2048])
        res_fc = tf.layers.dense(res_pool3_flat, res_nfc,
                                 activation = res_activation,
                                 name='res_fc')
        
        #
        res_logits = tf.layers.dense(res_fc, nout, name='res_output')
        return res_logits
    

