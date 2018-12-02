# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from skimage import transform
from joblib import Parallel, delayed
import cv2


from vgg import build_vgg
from inc import build_inc
from res import build_res

height = 64
width = 64

Y_dic = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Suprise',
         6:'Neutral'}
n_outputs = 7
font = cv2.FONT_HERSHEY_SIMPLEX


# import data
os.chdir('C:/ml/')
PATH = os.getcwd()
SCALE = 0.5
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0" 
os.environ["OPENCV_VIDEOIO_DEBUG"] = "1" 
#data = pd.read_csv('./fer2013.csv')


def process_image_from_pixel(pixel, target_width=width, target_height=height,
                  original_width = 48, original_height = 48, precision = np.float32):
    '''
    take a string representing pixels
    convert it to a np array
    '''
    face = np.array([int(i) for i in pixel.split(' ')],
                     dtype=precision).reshape((original_width, original_height))

    
#    # get target aspect ratio
#    original_ratio = original_width / original_height
#    target_ratio = target_width / target_height
#
#    # determine if we need to crop
#    too_high = original_ratio < target_ratio
#    crop_width = original_width if too_high else int(original_height * target_ratio)
#    crop_height = original_height if not too_high else int(original_width / target_ratio)
#
#    # zoom image
#    crop_width = int(crop_width)
#    crop_height = int(crop_height)
#
#    # select bounding box
#    
#    x_1 =  crop_width
#    y_1 =  crop_height
#
#    face = face[0:y_1, 0:x_1]

    face = face/255
    
    # resize to target dimension
    
    face = transform.resize(face, (target_width, target_height), 
                                     mode = 'constant',anti_aliasing=True)

    return face


def process_image_from_cv(face, target_width=width, target_height=height):

    face = face/255
    
    # resize to target dimension
    
    face = transform.resize(face, (target_width, target_height), 
                                     mode = 'constant',anti_aliasing=True)
    
    face = np.expand_dims(face,0)
    face = np.expand_dims(face,3)
    
    return face

def process_many_parallel(pixels):
    '''
    input: list(string)
    output: nparray of shape (n, target_wdith, target_height)
    '''
    
    res = Parallel(n_jobs=8)(delayed(process_image_from_pixel) (i) for i in pixels)
    X_batch = np.stack(res)
    return np.expand_dims(X_batch, 3)


################################################################################

tf.reset_default_graph()


with tf.name_scope('Input'):
    X = tf.placeholder(tf.float32, shape=[None, height, width,1])
    y = tf.placeholder(tf.int32, shape=[None], name='y')
    training = tf.placeholder_with_default(False, shape=[], name='training')



with tf.variable_scope('VGG_train'):
    vgg_logits = build_vgg(X_input=X, training=training,
                                nout = n_outputs)
    vgg_out = tf.stop_gradient(vgg_logits)
    vgg_prob = tf.nn.softmax(vgg_out)
    
#    vgg_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = vgg_out,
#                                                                  labels= y)
#    vgg_loss = tf.reduce_mean(vgg_xentropy)
#
#    vgg_correct = tf.nn.in_top_k(vgg_out, y, 1)
#    vgg_accuracy = tf.reduce_mean(tf.cast(vgg_correct, tf.float32))

with tf.variable_scope('Inception_train'):
    inc_logits = build_inc(X, n_outputs, training)
    inc_out = tf.stop_gradient(inc_logits)
    inc_prob = tf.nn.softmax(inc_out)
    
#    inc_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = inc_out,
#                                                                  labels= y)
#    inc_loss = tf.reduce_mean(inc_xentropy)
#    
#    inc_correct = tf.nn.in_top_k(inc_out, y, 1)
#    inc_accuracy = tf.reduce_mean(tf.cast(inc_correct, tf.float32))


with tf.variable_scope('ResNet_train'):
    # not parallel due to batch norm
    res_logits = build_res(X, training=training,nout = n_outputs)
    res_out = tf.stop_gradient(res_logits)
    res_prob = tf.nn.softmax(res_out)
#    res_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = res_out,
#                                                                  labels= y)
#    res_loss = tf.reduce_mean(res_xentropy)
#    
#    
#    res_correct = tf.nn.in_top_k(res_out, y, 1)
#    res_accuracy = tf.reduce_mean(tf.cast(res_correct, tf.float32))

with tf.name_scope('init_and_save'):
    init = tf.global_variables_initializer()
    saver_vgg = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='VGG_train'))
    saver_inc = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='Inception_train'))
    saver_res = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='ResNet_train'))
    saver_all = tf.train.Saver()
    
def avg(X_all,batch_size,sess):
    
    res = []
    for i in range(len(X_all)//batch_size):
        X_batch = X_all[i*batch_size:(i+1)*batch_size, :,:,:]
        vgg_probs = sess.run(vgg_prob,  feed_dict={X: X_batch} )
        inc_probs = sess.run(inc_prob,  feed_dict={X: X_batch} )
        res_probs = sess.run(res_prob,  feed_dict={X: X_batch} )
        
        tmp = np.zeros((batch_size,n_outputs))
        for i in range(batch_size):
            three = np.vstack([vgg_probs[i], inc_probs[i], res_probs[i]])
            tmp[i] = np.mean(three, axis=0)

        tmp= np.argmax(tmp,axis=1)
        
        res.append(tmp)
    return np.concatenate(res)

with tf.Session() as sess:
    init.run()
    
    saver_vgg.restore(sess, './vgg_v2/my_vgg')
    saver_inc.restore(sess, './inc_v2/my_inc')
    saver_res.restore(sess, './res_v2/my_res')
    
#    mydata = data[100:120]
#    Y_vali = mydata['emotion'].tolist()
#    X_raw = mydata['pixels'].tolist()
#    X_vali = process_many_parallel(X_raw)
#    
#    preds = avg(X_vali, 5, sess)
#    
#    for i in range(len(X_vali)):
#        pixel = X_raw[i]
#        face = np.array([int(j) for j in pixel.split(' ')]).reshape((48, 48))
#        plt.figure(figsize=(6, 6))
#        plt.imshow(face, cmap='gray')
#        plt.title('True: '+Y_dic[Y_vali[i]] +' , predicted: '+  Y_dic[preds[i]])
#        plt.axis("off")
#        plt.savefig('./pics/'+str(i)+'.png', bbox_inches='tight')
#        plt.close()
    
    
    face_cascade = cv2.CascadeClassifier('./cg/haarcascade_frontalface_default.xml')   # load the detection model
    cap=cv2.VideoCapture(0) # open camera
    cv2.startWindowThread()
    while True:
        ret,frame=cap.read()
        Img = frame
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            smallImg = cv2.resize(gray,(int(gray.shape[1]*SCALE),int(gray.shape[0]*SCALE)))
            smallImg = cv2.equalizeHist(smallImg)
            faces = face_cascade.detectMultiScale(smallImg, 1.3, 5)

            for face in faces:
                face = (face/SCALE).astype(int) 
                cv2.rectangle(frame, (face[0], face[1]), (face[0]+face[2],face[1]+face[3]), (255, 0, 0), 2)
                
                box = gray[face[1]:face[1]+face[3], face[0]:face[0]+face[2]] #
                box = process_image_from_cv(box)
                preds = avg(box,1,sess)
                cv2.putText(frame, Y_dic[preds[0]],
                            (face[0], face[1]),
                            font,1,(255,255,255),2)
                
            cv2.imshow("face_recognition",frame)
            if cv2.waitKey(40) & 0xff == ord("q"):
                break
  
    cap.release()
    cv2.destroyAllWindows()
    print('The camera is shut down.')
