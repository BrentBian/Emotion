import tensorflow as tf
import os
import timeit
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed
from skimage import transform

from vgg import build_vgg
from inc import build_inc
from res import build_res


height = 64
width = 64

##########################
#       Constants        #
##########################

Y_dic = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Suprise',
         6:'Neutral'}

height = 64
width = 64

n_outputs = 7
train_ratio = 0.8
vali_ratio = 0.9
np.random.seed(42)
##########################

# import data
os.chdir('L:/ml/facial')
MODEL_PATH = os.path.join('models')
data = pd.read_csv('./fer2013.csv')

# random shuffle
data = data.sample(frac=1).reset_index(drop=True)
training_raw = data[:int(len(data) * train_ratio)]
validation_raw = data[int(len(data) * train_ratio):int(len(data) * vali_ratio) ]
test_raw = data[int(len(data) * vali_ratio) :]

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

def process_image(pixel, target_width=width, target_height=height, max_zoom = 0.2,
                    original_width = 48, original_height = 48, precision = np.float32):
    '''
    take a string representing pixels
    convert it to a np array
    perform data augmantation
    '''
    face = np.array([int(i) for i in pixel.split(' ')],
                     dtype=precision).reshape((original_width, original_height))

    # get target aspect ratio
    original_ratio = original_width / original_height
    target_ratio = target_width / target_height

    # determine if we need to crop
    too_high = original_ratio < target_ratio
    crop_width = original_width if too_high else int(original_height * target_ratio)
    crop_height = original_height if not too_high else int(original_width / target_ratio)

    # zoom image
    zoom = np.random.rand() * max_zoom + 1
    crop_width = int(crop_width / zoom)
    crop_height = int(crop_height / zoom)

    # select bounding box
    x_0 = np.random.randint(0, original_width - crop_width)
    y_0 = np.random.randint(0, original_height - crop_height)
    
    x_1 = x_0 + crop_width
    y_1 = y_0 + crop_height

    face = face[y_0:y_1, x_0:x_1]

    # flip it with 50% probability
    if np.random.rand() > 0.5:
        face = np.fliplr(face)

    # rotate by a random angle
    ang = np.random.uniform(-10,10)
    face = face/255
    face = transform.rotate(face, ang)
    
    # resize to target dimension
    
    face = transform.resize(face, (target_width, target_height), 
                                     mode = 'constant',anti_aliasing=True)

    return face


def process_many(pixels):
    '''
    input: list(string)
    output: nparray of shape (n, target_wdith, target_height)
    '''

    res = [process_image(i) for i in pixels]
    X_batch = np.stack(res)
    return np.expand_dims(X_batch, 3)



def process_many_parallel(pixels):
    '''
    input: list(string)
    output: nparray of shape (n, target_wdith, target_height)
    '''
    
    res = Parallel(n_jobs=31)(delayed(process_image) (i) for i in pixels)
    X_batch = np.stack(res)
    return np.expand_dims(X_batch, 3)


##check above functions
#example = data[101:110]
#start = timeit.default_timer()
#processed = process_many(example['pixels'].tolist())
#emotions = example['emotion'].tolist()
#print('Time used:', timeit.default_timer()-start)
#for idx,i in enumerate(processed):
#    plt.figure(figsize=(6, 6))
#    plt.imshow(i[:,:,0], cmap='gray')
#    plt.title(Y_dic[emotions[idx]])
#    plt.axis("off")
#    plt.show()

#start = timeit.default_timer()    
#processed = process_many_parallel(example['pixels'].tolist())
#print('Time used:', timeit.default_timer()-start)
#for idx,i in enumerate(processed):
#    plt.figure(figsize=(6, 6))
#    plt.imshow(i[:,:,0], cmap='gray')
#    plt.title("{}x{}".format(i.shape[1], i.shape[0]))
#    plt.axis("off")
#    plt.show()
    

##################################################
# model building
##################################################
tf.reset_default_graph()


with tf.name_scope('Input'):
    X = tf.placeholder(tf.float32, shape=[None, height, width,1])
    y = tf.placeholder(tf.int32, shape=[None], name='y')
    training = tf.placeholder_with_default(False, shape=[], name='training')


def parallel_train(fun, num_gpu, nout, training, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpu)

    res = []
    for i in range(num_gpu):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                res.append(fun(nout=nout, training=training,
                               **{k : v[i] for k, v in in_splits.items()}))

    return tf.concat(res, axis=0)

with tf.variable_scope('VGG_train'):
    vgg_logits = parallel_train(build_vgg, 2, X_input=X, training=training,
                                nout = n_outputs)
    vgg_prob = tf.nn.softmax(vgg_logits)
    
    vgg_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = vgg_logits,
                                                                  labels= y)
    vgg_loss = tf.reduce_mean(vgg_xentropy)
    
    vgg_optimizer = tf.train.AdamOptimizer()
    vgg_training_op = vgg_optimizer.minimize(vgg_loss, colocate_gradients_with_ops=True)
    
    vgg_correct = tf.nn.in_top_k(vgg_logits, y, 1)
    vgg_accuracy = tf.reduce_mean(tf.cast(vgg_correct, tf.float32))

with tf.variable_scope('Inception_train'):
#    inc_logits = parallel_train(build_inc, 2, X_input=X, training=training,
#                                nout = n_outputs)
    inc_logits = build_inc(X, n_outputs, training)
    inc_prob = tf.nn.softmax(inc_logits)
    
    inc_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = inc_logits,
                                                                  labels= y)
    inc_loss = tf.reduce_mean(inc_xentropy)
    
    inc_optimizer = tf.train.AdamOptimizer()
    inc_training_op = inc_optimizer.minimize(inc_loss, colocate_gradients_with_ops=True)
    
    inc_correct = tf.nn.in_top_k(inc_logits, y, 1)
    inc_accuracy = tf.reduce_mean(tf.cast(inc_correct, tf.float32))



with tf.variable_scope('ResNet_train'):
    # not parallel due to batch norm
    res_logits = build_res(X, training=training,nout = n_outputs) 
    res_prob = tf.nn.softmax(res_logits)
    res_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = res_logits,
                                                                  labels= y)
    res_loss = tf.reduce_mean(res_xentropy)
    
    res_optimizer = tf.train.AdamOptimizer()
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        res_training_op = res_optimizer.minimize(res_loss)
    
    res_correct = tf.nn.in_top_k(res_logits, y, 1)
    res_accuracy = tf.reduce_mean(tf.cast(res_correct, tf.float32))
    

with tf.name_scope('init_and_save'):
    init = tf.global_variables_initializer()
    saver_vgg = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='VGG_train'))
    saver_inc = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='Inception_train'))
    saver_res = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='ResNet_train'))
    saver_all = tf.train.Saver()
    

def vgg_eval_accuracy_in_batches(X_all,Y_all,batch_size):
    res = []
    for i in range(len(X_all)//batch_size):
        X_batch = X_all[i*batch_size:(i+1)*batch_size, :,:,:]
        Y_batch = Y_all[i*batch_size:(i+1)*batch_size]
        tmp = sess.run(vgg_accuracy,  feed_dict={X: X_batch, y: Y_batch} )
        res.append(tmp)
    return np.mean(res)

def vgg_eval_loss_in_batches(X_all,Y_all,batch_size):
    res = []
    for i in range(len(X_all)//batch_size):
        X_batch = X_all[i*batch_size:(i+1)*batch_size, :,:,:]
        Y_batch = Y_all[i*batch_size:(i+1)*batch_size]
        tmp = sess.run(vgg_loss,  feed_dict={X: X_batch, y: Y_batch} )
        res.append(tmp)
    return np.mean(res)

def inc_eval_accuracy_in_batches(X_all,Y_all,batch_size):
    res = []
    for i in range(len(X_all)//batch_size):
        X_batch = X_all[i*batch_size:(i+1)*batch_size, :,:,:]
        Y_batch = Y_all[i*batch_size:(i+1)*batch_size]
        tmp = sess.run(inc_accuracy,  feed_dict={X: X_batch, y: Y_batch} )
        res.append(tmp)
    return np.mean(res)

def inc_eval_loss_in_batches(X_all,Y_all,batch_size):
    res = []
    for i in range(len(X_all)//batch_size):
        X_batch = X_all[i*batch_size:(i+1)*batch_size, :,:,:]
        Y_batch = Y_all[i*batch_size:(i+1)*batch_size]
        tmp = sess.run(inc_loss,  feed_dict={X: X_batch, y: Y_batch} )
        res.append(tmp)
    return np.mean(res)

def res_eval_accuracy_in_batches(X_all,Y_all,batch_size):
    res = []
    for i in range(len(X_all)//batch_size):
        X_batch = X_all[i*batch_size:(i+1)*batch_size, :,:,:]
        Y_batch = Y_all[i*batch_size:(i+1)*batch_size]
        tmp = sess.run(res_accuracy,  feed_dict={X: X_batch, y: Y_batch} )
        res.append(tmp)
    return np.mean(res)

def res_eval_loss_in_batches(X_all,Y_all,batch_size):
    res = []
    for i in range(len(X_all)//batch_size):
        X_batch = X_all[i*batch_size:(i+1)*batch_size, :,:,:]
        Y_batch = Y_all[i*batch_size:(i+1)*batch_size]
        tmp = sess.run(res_loss,  feed_dict={X: X_batch, y: Y_batch} )
        res.append(tmp)
    return np.mean(res)

    
def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#    for gvar, value in zip(gvars, tf.get_default_session().run(gvars)):
#        print(gvar)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)
    


#############################


n_epochs = 1000
batch_size = 1600

best_loss_val = np.infty
check_interval = 10
checks_since_last_progress = 0
max_checks_without_progress = 50

best_model_params = None 

#file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess:
    init.run()
    
#    saver_all.restore(sess, './my_all')
    saver_vgg.restore(sess, './vgg_v2/my_vgg')
#    saver_inc.restore(sess, './inc_v2/my_inc')
#    saver_res.restore(sess, './res_v2/my_res')
    
    start_time = timeit.default_timer()
    Y_vali = validation_raw['emotion'].tolist()
    X_vali = process_many_parallel(validation_raw['pixels'].tolist())
    
    for epoch in range(n_epochs):
        this_time = timeit.default_timer()
        for iteration in range(int(len(data) * train_ratio) // batch_size):
            train_batch_raw = training_raw.sample(n=batch_size, replace=True)
            Y_batch = train_batch_raw['emotion'].tolist()
            X_batch = process_many_parallel(train_batch_raw['pixels'].tolist())
            sess.run(vgg_training_op, feed_dict={X: X_batch, y: Y_batch, training: True})
            if iteration % check_interval == 0:
                
                loss_val = vgg_eval_loss_in_batches(X_vali, Y_vali, batch_size)
#                acc_val = vgg_eval_accuracy_in_batches(X_vali,Y_vali, batch_size)
#                print('Current validation accuracy: {:.4f}%'.format(acc_val*100))
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
                    
#                step = epoch * int(len(data) * train_ratio) // batch_size + iteration
#                summary_str = vgg_accuracy_summary.eval(feed_dict={X: X_batch, y: Y_batch})
#                file_writer.add_summary(summary_str, step)
        acc_train = vgg_accuracy.eval(feed_dict={X: X_batch, y: Y_batch})
        acc_val = vgg_eval_accuracy_in_batches(X_vali,Y_vali, batch_size)
        print("Epoch {}, train accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}, time per epoch: {:.4f}".format(
                  epoch, acc_train * 100, acc_val * 100, best_loss_val, timeit.default_timer()-this_time))
        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break

    if best_model_params:
        restore_model_params(best_model_params)
        
    Y_test = test_raw['emotion'].tolist()
    X_test = process_many_parallel(test_raw['pixels'].tolist())
    acc_test = vgg_eval_accuracy_in_batches(X_test,Y_test, batch_size)
    print("Final accuracy on test set: {:.4f}%, total time: {:.2f}".format(acc_test*100,
          timeit.default_timer()- start_time))
    saver_vgg.save(sess, "./my_vgg")
    saver_inc.save(sess, "./my_inc")
    saver_res.save(sess, "./my_res")
    saver_all.save(sess, "./my_all")
    
#    
#
#file_writer.flush()
#file_writer.close()
