# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:54:37 2018

@author: brent
"""
import tensorflow as tf

# definitions for vgg16



vgg_drop_rate = 0.82
vgg_nfc1 = 2048
vgg_nfc2 = 2048
vgg_nfc3 = 1024

vgg_norm_r = 2
vgg_norm_a = 0.00002
vgg_norm_beta = 0.75
vgg_norm_bias = 1




vgg_conv1_fmaps = 64
vgg_conv1_ksize = 3
vgg_conv1_stride = 1
vgg_conv1_pad = 'SAME'

vgg_conv2_fmaps = 64
vgg_conv2_ksize = 3
vgg_conv2_stride = 1
vgg_conv2_pad = 'SAME'

vgg_pool3_ksize = [1,3,3,1]
vgg_pool3_stride = [1,2,2,1]
vgg_pool3_pad = 'SAME'


#########

vgg_conv4_fmaps = 128
vgg_conv4_ksize = 3
vgg_conv4_stride = 1
vgg_conv4_pad = 'SAME'

vgg_conv5_fmaps = 128
vgg_conv5_ksize = 3
vgg_conv5_stride = 1
vgg_conv5_pad = 'SAME'

vgg_pool6_ksize = [1,3,3,1]
vgg_pool6_stride = [1,2,2,1]
vgg_pool6_pad = 'SAME'


#########
vgg_conv7_fmaps = 256
vgg_conv7_ksize = 3
vgg_conv7_stride = 1
vgg_conv7_pad = 'SAME'

vgg_conv8_fmaps = 256
vgg_conv8_ksize = 3
vgg_conv8_stride = 1
vgg_conv8_pad = 'SAME'

vgg_pool9_ksize = [1,3,3,1]
vgg_pool9_stride = [1,2,2,1]
vgg_pool9_pad = 'SAME'

#########
vgg_conv10_fmaps = 512
vgg_conv10_ksize = 3
vgg_conv10_stride = 1
vgg_conv10_pad = 'SAME'

vgg_conv11_fmaps = 512
vgg_conv11_ksize = 3
vgg_conv11_stride = 1
vgg_conv11_pad = 'SAME'

vgg_conv12_fmaps = 512
vgg_conv12_ksize = 3
vgg_conv12_stride = 1
vgg_conv12_pad = 'SAME'

vgg_pool13_ksize = [1,3,3,1]
vgg_pool13_stride = [1,2,2,1]
vgg_pool13_pad = 'SAME'

#########
vgg_conv14_fmaps = 512
vgg_conv14_ksize = 3
vgg_conv14_stride = 1
vgg_conv14_pad = 'SAME'

vgg_conv15_fmaps = 512
vgg_conv15_ksize = 3
vgg_conv15_stride = 1
vgg_conv15_pad = 'SAME'

vgg_conv16_fmaps = 512
vgg_conv16_ksize = 3
vgg_conv16_stride = 1
vgg_conv16_pad = 'SAME'

vgg_pool17_fmaps = 512
vgg_pool17_ksize = [1,3,3,1]
vgg_pool17_stride = [1,2,2,1]
vgg_pool17_pad = 'SAME'



vgg_activation = tf.nn.elu

#################################
#  end of vgg definitions
#################################


def build_vgg(X_input, training, nout):
    with tf.name_scope('VGG'):
        vgg_conv1 = tf.layers.conv2d(X_input, filters = vgg_conv1_fmaps,
                                     kernel_size=vgg_conv1_ksize,
                                     strides = vgg_conv1_stride, padding = vgg_conv1_pad,
                                     activation = vgg_activation, name = 'vgg_conv1')
        vgg_conv2 = tf.layers.conv2d(vgg_conv1, filters = vgg_conv2_fmaps,
                                     kernel_size=vgg_conv2_ksize,
                                     strides = vgg_conv2_stride, padding = vgg_conv2_pad,
                                     activation = vgg_activation, name = 'vgg_conv2')
        
        vgg_pool3 = tf.nn.max_pool(vgg_conv2, ksize= vgg_pool3_ksize, strides= vgg_pool3_stride,
                               padding = vgg_pool3_pad, name='vgg_pool1')
        vgg_norm1 = tf.nn.local_response_normalization(vgg_pool3,
                                                       depth_radius=vgg_norm_r,
                                                       bias=vgg_norm_bias,
                                                       alpha=vgg_norm_a,
                                               beta=vgg_norm_beta, name='vgg_norm1')
        # 
        
        vgg_conv4 = tf.layers.conv2d(vgg_norm1, filters = vgg_conv4_fmaps,
                                     kernel_size=vgg_conv4_ksize,
                                     strides = vgg_conv4_stride, padding = vgg_conv4_pad,
                                     activation = vgg_activation, name = 'vgg_conv4')
        vgg_conv5 = tf.layers.conv2d(vgg_conv4, filters = vgg_conv5_fmaps,
                                     kernel_size=vgg_conv5_ksize,
                                     strides = vgg_conv5_stride, padding = vgg_conv5_pad,
                                     activation = vgg_activation, name = 'vgg_conv5')
        
        vgg_pool6 = tf.nn.max_pool(vgg_conv5, ksize= vgg_pool6_ksize, strides= vgg_pool6_stride,
                               padding = vgg_pool6_pad, name='vgg_pool6')
    
        vgg_norm2 = tf.nn.local_response_normalization(vgg_pool6,
                                                       depth_radius=vgg_norm_r,
                                                       bias=vgg_norm_bias,
                                                       alpha=vgg_norm_a,
                                               beta=vgg_norm_beta, name='vgg_norm2')
        #
        
        vgg_conv7 = tf.layers.conv2d(vgg_norm2 , filters = vgg_conv7_fmaps,
                                     kernel_size=vgg_conv7_ksize,
                                     strides = vgg_conv7_stride, padding = vgg_conv7_pad,
                                     activation = vgg_activation, name = 'vgg_conv7')
        vgg_conv8 = tf.layers.conv2d(vgg_conv7, filters = vgg_conv8_fmaps,
                                     kernel_size=vgg_conv8_ksize,
                                     strides = vgg_conv8_stride, padding = vgg_conv8_pad,
                                     activation = vgg_activation, name = 'vgg_conv8')
        
        vgg_pool9 = tf.nn.max_pool(vgg_conv8, ksize= vgg_pool9_ksize, strides= vgg_pool9_stride,
                               padding = vgg_pool9_pad, name='vgg_pool9')
        vgg_norm3 = tf.nn.local_response_normalization(vgg_pool9,
                                                       depth_radius=vgg_norm_r,
                                                       bias=vgg_norm_bias,
                                                       alpha=vgg_norm_a,
                                               beta=vgg_norm_beta, name='vgg_norm3')
        
        #
        vgg_conv10 = tf.layers.conv2d(vgg_norm3, filters = vgg_conv10_fmaps,
                                     kernel_size=vgg_conv10_ksize,
                                     strides = vgg_conv10_stride, padding = vgg_conv10_pad,
                                     activation = vgg_activation, name = 'vgg_conv10')
        vgg_conv11 = tf.layers.conv2d(vgg_conv10, filters = vgg_conv11_fmaps,
                                     kernel_size=vgg_conv11_ksize,
                                     strides = vgg_conv1_stride, padding = vgg_conv11_pad,
                                     activation = vgg_activation, name = 'vgg_conv11')
        vgg_conv12 = tf.layers.conv2d(vgg_conv11, filters = vgg_conv12_fmaps,
                                     kernel_size=vgg_conv12_ksize,
                                     strides = vgg_conv12_stride, padding = vgg_conv12_pad,
                                     activation = vgg_activation, name = 'vgg_conv12')
        
        vgg_pool13 = tf.nn.max_pool(vgg_conv12, ksize= vgg_pool13_ksize, strides= vgg_pool13_stride,
                               padding = vgg_pool13_pad, name='vgg_pool13')
        
        vgg_norm4 = tf.nn.local_response_normalization(vgg_pool13,
                                                       depth_radius=vgg_norm_r,
                                                       bias=vgg_norm_bias,
                                                       alpha=vgg_norm_a,
                                               beta=vgg_norm_beta, name='vgg_norm4')
        vgg_norm4_drop = tf.layers.dropout(vgg_norm4, rate=vgg_drop_rate, training=training)
        
        #
        vgg_conv14 = tf.layers.conv2d(vgg_norm4_drop, filters = vgg_conv14_fmaps,
                                     kernel_size=vgg_conv14_ksize,
                                     strides = vgg_conv14_stride, padding = vgg_conv14_pad,
                                     activation = vgg_activation, name = 'vgg_conv14')
        vgg_conv15 = tf.layers.conv2d(vgg_conv14, filters = vgg_conv15_fmaps,
                                     kernel_size=vgg_conv15_ksize,
                                     strides = vgg_conv15_stride, padding = vgg_conv15_pad,
                                     activation = vgg_activation, name = 'vgg_conv15')
        vgg_conv16 = tf.layers.conv2d(vgg_conv15, filters = vgg_conv16_fmaps,
                                     kernel_size=vgg_conv16_ksize,
                                     strides = vgg_conv16_stride, padding = vgg_conv16_pad,
                                     activation = vgg_activation, name = 'vgg_conv16')
        
        vgg_pool17 = tf.nn.max_pool(vgg_conv16, ksize= vgg_pool17_ksize, strides= vgg_pool17_stride,
                               padding = vgg_pool17_pad, name='vgg_pool17')
        vgg_norm5 = tf.nn.local_response_normalization(vgg_pool17,
                                                       depth_radius=vgg_norm_r,
                                                       bias=vgg_norm_bias,
                                                       alpha=vgg_norm_a,
                                               beta=vgg_norm_beta, name='vgg_norm5')
        #
        vgg_pool17_flat = tf.reshape(vgg_norm5, shape=[-1, vgg_pool17_fmaps*4])
        vgg_pool17_flat_drop = tf.layers.dropout(vgg_pool17_flat,rate=vgg_drop_rate, training=training)
        vgg_fc1 = tf.layers.dense(vgg_pool17_flat_drop, vgg_nfc1, activation=vgg_activation)
        vgg_fc1_drop = tf.layers.dropout(vgg_fc1, rate=vgg_drop_rate, training=training)
        vgg_fc2 = tf.layers.dense(vgg_fc1_drop, vgg_nfc2, activation=vgg_activation)
        vgg_fc2_drop = tf.layers.dropout(vgg_fc2, rate=vgg_drop_rate, training=training)
        vgg_fc3 = tf.layers.dense(vgg_fc2_drop, vgg_nfc3, activation=vgg_activation,
                                  name='vgg_fc3')
        
        #
        vgg_logits = tf.layers.dense(vgg_fc3, nout, name='vgg_output')
        return vgg_logits
    


#################################
#  end of VGG definitions
#################################