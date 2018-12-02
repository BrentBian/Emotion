# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:57:27 2018

@author: brent
"""

import tensorflow as tf


# definitions for Inception V2

# begin of stem
inc_conv1_fmaps = 64
inc_conv1_ksize = 3
inc_conv1_stride = 1
inc_conv1_pad = 'SAME'

inc_pool2_ksize = [1,3,3,1]
inc_pool2_stride = [1,2,2,1]
inc_pool2_pad = 'SAME'

inc_conv3_fmaps = 64
inc_conv3_ksize = 1
inc_conv3_stride = 1
inc_conv3_pad = 'SAME'

inc_conv4_fmaps = 192
inc_conv4_ksize = 3
inc_conv4_stride = 1
inc_conv4_pad = 'SAME'

inc_pool5_ksize = [1,3,3,1]
inc_pool5_stride = [1,2,2,1]
inc_pool5_pad = 'SAME'
# end of stem

# begin of inception module
inc_mod_p1_ksize = 1

inc_mod_p2_1_ksize = 1
inc_mod_p2_2_ksize = 3

inc_mod_p3_1_ksize = 1
inc_mod_p3_2_ksize = 5

inc_mod_p4_1_ksize = [1,3,3,1]
inc_mod_p4_2_ksize = 1
# endof inception module

inc_pool8_ksize = [1,3,3,1]
inc_pool8_stride = [1,2,2,1]
inc_pool8_pad = 'SAME'

inc_pool14_ksize = [1,3,3,1]
inc_pool14_stride = [1,2,2,1]
inc_pool14_pad = 'SAME'

inc_pool17_ksize = [1,4,4,1]
inc_pool17_stride = [1,1,1,1]
inc_pool17_pad = 'VALID'


inc_drop_rate = 0.65
inc_nfc = 1000

inc_activation = tf.nn.relu

inc_norm_r = 2
inc_norm_a = 0.00002
inc_norm_beta = 0.75
inc_norm_bias = 1


#################################
#  end of Inception definitions
#################################



def build_inc(X_input, nout, training):
    with tf.name_scope('Inception'):
        inc_conv1 = tf.layers.conv2d(X_input, filters = inc_conv1_fmaps,
                                     kernel_size = inc_conv1_ksize,
                                     strides = inc_conv1_stride,
                                     padding = inc_conv1_pad,
                                     activation = inc_activation,
                                     name = 'inc_cov1')
        
        inc_pool2 = tf.nn.max_pool(inc_conv1, ksize = inc_pool2_ksize,
                                   strides = inc_pool2_stride,
                                   padding = inc_pool2_pad,
                                   name = 'inc_pool2')
        
        inc_norm1 = tf.nn.local_response_normalization(inc_pool2,
                                                       depth_radius=inc_norm_r,
                                                       bias = inc_norm_bias,
                                                       alpha = inc_norm_a,
                                                       beta = inc_norm_beta,
                                                       name = 'inc_norm1')
        
        inc_conv3 = tf.layers.conv2d(inc_norm1, filters = inc_conv3_fmaps,
                                     kernel_size = inc_conv3_ksize,
                                     strides = inc_conv3_stride,
                                     padding = inc_conv3_pad,
                                     activation = inc_activation,
                                     name = 'inc_cov3')
        
        inc_conv4 = tf.layers.conv2d(inc_conv3, filters = inc_conv4_fmaps,
                                     kernel_size = inc_conv4_ksize,
                                     strides = inc_conv4_stride,
                                     padding = inc_conv4_pad,
                                     activation = inc_activation,
                                     name = 'inc_cov4')
        
        inc_norm2 = tf.nn.local_response_normalization(inc_conv4,
                                                       depth_radius=inc_norm_r,
                                                       bias = inc_norm_bias,
                                                       alpha = inc_norm_a,
                                                       beta = inc_norm_beta,
                                                       name = 'inc_norm2')
        
        inc_pool5 = tf.nn.max_pool(inc_norm2, ksize = inc_pool5_ksize,
                                   strides = inc_pool5_stride,
                                   padding = inc_pool5_pad,
                                   name = 'inc_pool5')
        
        # Inception Module NO.1
        inc_mod1_p1 = tf.layers.conv2d(inc_pool5, filters = 64,
                                     kernel_size = inc_mod_p1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod1_p2_1 = tf.layers.conv2d(inc_pool5, filters = 96,
                                     kernel_size = inc_mod_p2_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod1_p2_2 = tf.layers.conv2d(inc_mod1_p2_1, filters = 128,
                                     kernel_size = inc_mod_p2_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod1_p3_1 = tf.layers.conv2d(inc_pool5, filters = 16,
                                     kernel_size = inc_mod_p3_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod1_p3_2 = tf.layers.conv2d(inc_mod1_p3_1, filters = 32,
                                     kernel_size = inc_mod_p3_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod1_p4_1 = tf.nn.max_pool(inc_pool5,
                                       ksize = inc_mod_p4_1_ksize,
                                       strides = [1,1,1,1],
                                       padding = 'SAME')
        
        inc_mod1_p4_2 = tf.layers.conv2d(inc_mod1_p4_1, filters = 32,
                                     kernel_size = inc_mod_p4_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod1_out = tf.concat((inc_mod1_p1, inc_mod1_p2_2, inc_mod1_p3_2,
                                  inc_mod1_p4_2), axis = 3)
        
        # Inception Module NO.2
        inc_mod2_p1 = tf.layers.conv2d(inc_mod1_out, filters = 128,
                                     kernel_size = inc_mod_p1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod2_p2_1 = tf.layers.conv2d(inc_mod1_out, filters = 128,
                                     kernel_size = inc_mod_p2_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod2_p2_2 = tf.layers.conv2d(inc_mod2_p2_1, filters = 192,
                                     kernel_size = inc_mod_p2_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod2_p3_1 = tf.layers.conv2d(inc_mod1_out, filters = 32,
                                     kernel_size = inc_mod_p3_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod2_p3_2 = tf.layers.conv2d(inc_mod2_p3_1, filters = 96,
                                     kernel_size = inc_mod_p3_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod2_p4_1 = tf.nn.max_pool(inc_mod1_out,
                                       ksize = inc_mod_p4_1_ksize,
                                       strides =  [1,1,1,1],
                                       padding = 'SAME')
        
        inc_mod2_p4_2 = tf.layers.conv2d(inc_mod2_p4_1, filters = 64,
                                     kernel_size = inc_mod_p4_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod2_out = tf.concat((inc_mod2_p1, inc_mod2_p2_2, inc_mod2_p3_2,
                                  inc_mod2_p4_2), axis = 3)
        
        
        inc_pool8 = tf.nn.max_pool(inc_mod2_out, ksize = inc_pool8_ksize,
                                   strides = inc_pool8_stride,
                                   padding = inc_pool8_pad,
                                   name = 'inc_pool8')
        
        # Inception Module NO.3
        inc_mod3_p1 = tf.layers.conv2d(inc_pool8, filters = 192,
                                     kernel_size = inc_mod_p1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod3_p2_1 = tf.layers.conv2d(inc_pool8, filters = 96,
                                     kernel_size = inc_mod_p2_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod3_p2_2 = tf.layers.conv2d(inc_mod3_p2_1, filters = 208,
                                     kernel_size = inc_mod_p2_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod3_p3_1 = tf.layers.conv2d(inc_pool8, filters = 16,
                                     kernel_size = inc_mod_p3_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod3_p3_2 = tf.layers.conv2d(inc_mod3_p3_1, filters = 48,
                                     kernel_size = inc_mod_p3_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod3_p4_1 = tf.nn.max_pool(inc_pool8,
                                       ksize = inc_mod_p4_1_ksize,
                                       strides =  [1,1,1,1],
                                       padding = 'SAME')
        
        inc_mod3_p4_2 = tf.layers.conv2d(inc_mod3_p4_1, filters = 64,
                                     kernel_size = inc_mod_p4_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod3_out = tf.concat((inc_mod3_p1, inc_mod3_p2_2, inc_mod3_p3_2,
                                  inc_mod3_p4_2), axis = 3)
        
        # Inception Module NO.4
        inc_mod4_p1 = tf.layers.conv2d(inc_mod3_out, filters = 160,
                                     kernel_size = inc_mod_p1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod4_p2_1 = tf.layers.conv2d(inc_mod3_out, filters = 112,
                                     kernel_size = inc_mod_p2_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod4_p2_2 = tf.layers.conv2d(inc_mod4_p2_1, filters = 224,
                                     kernel_size = inc_mod_p2_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod4_p3_1 = tf.layers.conv2d(inc_mod3_out, filters = 24,
                                     kernel_size = inc_mod_p3_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod4_p3_2 = tf.layers.conv2d(inc_mod4_p3_1, filters = 64,
                                     kernel_size = inc_mod_p3_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod4_p4_1 = tf.nn.max_pool(inc_mod3_out,
                                       ksize = inc_mod_p4_1_ksize,
                                       strides =  [1,1,1,1],
                                       padding = 'SAME')
        
        inc_mod4_p4_2 = tf.layers.conv2d(inc_mod4_p4_1, filters = 64,
                                     kernel_size = inc_mod_p4_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod4_out = tf.concat((inc_mod4_p1, inc_mod4_p2_2, inc_mod4_p3_2,
                                  inc_mod4_p4_2), axis = 3)
        
        
        # Inception Module NO.5
        inc_mod5_p1 = tf.layers.conv2d(inc_mod4_out, filters = 128,
                                     kernel_size = inc_mod_p1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod5_p2_1 = tf.layers.conv2d(inc_mod4_out, filters = 128,
                                     kernel_size = inc_mod_p2_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod5_p2_2 = tf.layers.conv2d(inc_mod5_p2_1, filters = 256,
                                     kernel_size = inc_mod_p2_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod5_p3_1 = tf.layers.conv2d(inc_mod4_out, filters = 24,
                                     kernel_size = inc_mod_p3_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod5_p3_2 = tf.layers.conv2d(inc_mod5_p3_1, filters = 64,
                                     kernel_size = inc_mod_p3_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod5_p4_1 = tf.nn.max_pool(inc_mod4_out,
                                       ksize = inc_mod_p4_1_ksize,
                                       strides =  [1,1,1,1],
                                       padding = 'SAME')
        
        inc_mod5_p4_2 = tf.layers.conv2d(inc_mod5_p4_1, filters = 64,
                                     kernel_size = inc_mod_p4_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod5_out = tf.concat((inc_mod5_p1, inc_mod5_p2_2, inc_mod5_p3_2,
                                  inc_mod5_p4_2), axis = 3)
        
        
         # Inception Module NO.6
        inc_mod6_p1 = tf.layers.conv2d(inc_mod5_out, filters = 112,
                                     kernel_size = inc_mod_p1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod6_p2_1 = tf.layers.conv2d(inc_mod5_out, filters = 144,
                                     kernel_size = inc_mod_p2_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod6_p2_2 = tf.layers.conv2d(inc_mod6_p2_1, filters = 288,
                                     kernel_size = inc_mod_p2_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod6_p3_1 = tf.layers.conv2d(inc_mod5_out, filters = 32,
                                     kernel_size = inc_mod_p3_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod6_p3_2 = tf.layers.conv2d(inc_mod6_p3_1, filters = 64,
                                     kernel_size = inc_mod_p3_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod6_p4_1 = tf.nn.max_pool(inc_mod5_out,
                                       ksize = inc_mod_p4_1_ksize,
                                       strides =  [1,1,1,1],
                                       padding = 'SAME')
        
        inc_mod6_p4_2 = tf.layers.conv2d(inc_mod6_p4_1, filters = 64,
                                     kernel_size = inc_mod_p4_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod6_out = tf.concat((inc_mod6_p1, inc_mod6_p2_2, inc_mod6_p3_2,
                                  inc_mod6_p4_2), axis = 3)
        
        # Inception Module NO.7
        inc_mod7_p1 = tf.layers.conv2d(inc_mod6_out, filters = 256,
                                     kernel_size = inc_mod_p1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod7_p2_1 = tf.layers.conv2d(inc_mod6_out, filters = 160,
                                     kernel_size = inc_mod_p2_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod7_p2_2 = tf.layers.conv2d(inc_mod7_p2_1, filters = 320,
                                     kernel_size = inc_mod_p2_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod7_p3_1 = tf.layers.conv2d(inc_mod6_out, filters = 32,
                                     kernel_size = inc_mod_p3_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod7_p3_2 = tf.layers.conv2d(inc_mod7_p3_1, filters = 128,
                                     kernel_size = inc_mod_p3_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod7_p4_1 = tf.nn.max_pool(inc_mod6_out,
                                       ksize = inc_mod_p4_1_ksize,
                                       strides =  [1,1,1,1],
                                       padding = 'SAME')
        
        inc_mod7_p4_2 = tf.layers.conv2d(inc_mod7_p4_1, filters = 128,
                                     kernel_size = inc_mod_p4_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod7_out = tf.concat((inc_mod7_p1, inc_mod7_p2_2, inc_mod7_p3_2,
                                  inc_mod7_p4_2), axis = 3)
    
    
        inc_pool14 = tf.nn.max_pool(inc_mod7_out, ksize = inc_pool14_ksize,
                                   strides = inc_pool14_stride,
                                   padding = inc_pool14_pad,
                                   name = 'inc_pool14')
        
        
        # Inception Module NO.8
        inc_mod8_p1 = tf.layers.conv2d(inc_pool14, filters = 256,
                                     kernel_size = inc_mod_p1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod8_p2_1 = tf.layers.conv2d(inc_pool14, filters = 160,
                                     kernel_size = inc_mod_p2_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod8_p2_2 = tf.layers.conv2d(inc_mod8_p2_1, filters = 320,
                                     kernel_size = inc_mod_p2_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod8_p3_1 = tf.layers.conv2d(inc_pool14, filters = 32,
                                     kernel_size = inc_mod_p3_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod8_p3_2 = tf.layers.conv2d(inc_mod8_p3_1, filters = 128,
                                     kernel_size = inc_mod_p3_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod8_p4_1 = tf.nn.max_pool(inc_pool14,
                                       ksize = inc_mod_p4_1_ksize,
                                       strides =  [1,1,1,1],
                                       padding = 'SAME')
        
        inc_mod8_p4_2 = tf.layers.conv2d(inc_mod8_p4_1, filters = 128,
                                     kernel_size = inc_mod_p4_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod8_out = tf.concat((inc_mod8_p1, inc_mod8_p2_2, inc_mod8_p3_2,
                                  inc_mod8_p4_2), axis = 3)
        
        # Inception Module NO.9
        inc_mod9_p1 = tf.layers.conv2d(inc_mod8_out, filters = 384,
                                     kernel_size = inc_mod_p1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod9_p2_1 = tf.layers.conv2d(inc_mod8_out, filters = 192,
                                     kernel_size = inc_mod_p2_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod9_p2_2 = tf.layers.conv2d(inc_mod9_p2_1, filters = 384,
                                     kernel_size = inc_mod_p2_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod9_p3_1 = tf.layers.conv2d(inc_mod8_out, filters = 48,
                                     kernel_size = inc_mod_p3_1_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        inc_mod9_p3_2 = tf.layers.conv2d(inc_mod9_p3_1, filters = 128,
                                     kernel_size = inc_mod_p3_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        
        
        inc_mod9_p4_1 = tf.nn.max_pool(inc_mod8_out,
                                       ksize = inc_mod_p4_1_ksize,
                                       strides =  [1,1,1,1],
                                       padding = 'SAME')
        
        inc_mod9_p4_2 = tf.layers.conv2d(inc_mod9_p4_1, filters = 128,
                                     kernel_size = inc_mod_p4_2_ksize,
                                     strides = 1,
                                     padding = 'SAME',
                                     activation = inc_activation)
        
        inc_mod9_out = tf.concat((inc_mod9_p1, inc_mod9_p2_2, inc_mod9_p3_2,
                                  inc_mod9_p4_2), axis = 3)
        
        
        inc_pool17 = tf.nn.avg_pool(inc_mod9_out, ksize = inc_pool17_ksize,
                                   strides = inc_pool17_stride,
                                   padding = inc_pool17_pad,
                                   name = 'inc_pool17')
        inc_drop = tf.layers.dropout(inc_pool17, rate = inc_drop_rate,
                                     training=training)
        
        inc_drop_flat = tf.reshape(inc_drop, shape=[-1, 1024])
        inc_fc = tf.layers.dense(inc_drop_flat, inc_nfc, activation=inc_activation,
                                 name = 'inc_fc')
        
        #
        inc_logits = tf.layers.dense(inc_fc, nout, name='inc_output')
        return inc_logits
    
    


#################################
#  end of Inception definitions
#################################