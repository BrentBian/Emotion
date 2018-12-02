import tensorflow as tf
import os
import timeit
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed
from skimage import transform
import seaborn as sn

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



n_epochs = 1000
batch_size = 500

best_loss_val = np.infty
check_interval = 10
checks_since_last_progress = 0
max_checks_without_progress = 100

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
    ang = np.random.uniform(-5,5)
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
#example = data[101:150]
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
#    plt.axis("off")
##    plt.savefig('./pics/p'+str(idx)+'.png', bbox_inches='tight')
#    plt.show()
#    

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
    vgg_out = tf.stop_gradient(vgg_logits)
    vgg_prob = tf.nn.softmax(vgg_out)
    
    vgg_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = vgg_out,
                                                                  labels= y)
    vgg_loss = tf.reduce_mean(vgg_xentropy)
    
#    vgg_optimizer = tf.train.AdamOptimizer()
#    vgg_training_op = vgg_optimizer.minimize(vgg_loss, colocate_gradients_with_ops=True)
    
    vgg_correct = tf.nn.in_top_k(vgg_out, y, 1)
    vgg_accuracy = tf.reduce_mean(tf.cast(vgg_correct, tf.float32))

with tf.variable_scope('Inception_train'):
#    inc_logits = parallel_train(build_inc, 2, X_input=X, training=training,
#                                nout = n_outputs)
    # single GPU actually faster
    with tf.device('/gpu:1'):
        inc_logits = build_inc(X, n_outputs, training)
    inc_out = tf.stop_gradient(inc_logits)
    inc_prob = tf.nn.softmax(inc_out)
    
    inc_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = inc_out,
                                                                  labels= y)
    inc_loss = tf.reduce_mean(inc_xentropy)
    
#    inc_optimizer = tf.train.AdamOptimizer()
#    inc_training_op = inc_optimizer.minimize(inc_loss, colocate_gradients_with_ops=True)
    
    inc_correct = tf.nn.in_top_k(inc_out, y, 1)
    inc_accuracy = tf.reduce_mean(tf.cast(inc_correct, tf.float32))


with tf.variable_scope('ResNet_train'):
    # not parallel due to batch norm
    with tf.device('/gpu:0'):
        res_logits = build_res(X, training=training,nout = n_outputs)
    res_out = tf.stop_gradient(res_logits)
    res_prob = tf.nn.softmax(res_out)
    res_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = res_out,
                                                                  labels= y)
    res_loss = tf.reduce_mean(res_xentropy)
    
    res_optimizer = tf.train.AdamOptimizer()
    
#    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#    with tf.control_dependencies(update_ops):
#        res_training_op = res_optimizer.minimize(res_loss)
#    
    res_correct = tf.nn.in_top_k(res_out, y, 1)
    res_accuracy = tf.reduce_mean(tf.cast(res_correct, tf.float32))
    


with tf.variable_scope('Pooling'):
    three = tf.concat((vgg_out,inc_out,res_out),axis=1)
    pool_logits = tf.layers.dense(three,n_outputs)
    pool_prob = tf.nn.softmax(pool_logits)
    pool_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pool_logits,
                                                                  labels= y)
    pool_loss = tf.reduce_mean(pool_xentropy)
    
    pool_optimizer = tf.train.AdamOptimizer()
    pool_training_op = pool_optimizer.minimize(pool_loss,colocate_gradients_with_ops=True)
    pool_correct = tf.nn.in_top_k(pool_prob,y,1)
    pool_accuracy = tf.reduce_mean(tf.cast(pool_correct,tf.float32))
    pool_pred = tf.argmax(pool_prob,axis=1)

with tf.name_scope('init_and_save'):
    init = tf.global_variables_initializer()
    saver_vgg = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='VGG_train'))
    saver_inc = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='Inception_train'))
    saver_res = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='ResNet_train'))
    saver_pool = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='Pooling'))
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

def pool_eval_accuracy_in_batches(X_all,Y_all,batch_size):
    res = []
    for i in range(len(X_all)//batch_size):
        X_batch = X_all[i*batch_size:(i+1)*batch_size, :,:,:]
        Y_batch = Y_all[i*batch_size:(i+1)*batch_size]
        tmp = sess.run(pool_accuracy,  feed_dict={X: X_batch, y: Y_batch} )
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

def pool_eval_loss_in_batches(X_all,Y_all,batch_size):
    res = []
    for i in range(len(X_all)//batch_size):
        X_batch = X_all[i*batch_size:(i+1)*batch_size, :,:,:]
        Y_batch = Y_all[i*batch_size:(i+1)*batch_size]
        tmp = sess.run(pool_loss,  feed_dict={X: X_batch, y: Y_batch} )
        res.append(tmp)
    return np.mean(res)

def pool_pred_in_batches(X_all,Y_all,batch_size):
    res = []
    for i in range(len(X_all)//batch_size):
        X_batch = X_all[i*batch_size:(i+1)*batch_size, :,:,:]
        Y_batch = Y_all[i*batch_size:(i+1)*batch_size]
        tmp = sess.run(pool_pred,  feed_dict={X: X_batch, y: Y_batch} )
        res.append(tmp)
    return tf.concat(res,axis=0)

def voting(X_all,Y_all,batch_size):
    
    res = []
    for i in range(len(X_all)//batch_size):
        X_batch = X_all[i*batch_size:(i+1)*batch_size, :,:,:]
        Y_batch = Y_all[i*batch_size:(i+1)*batch_size]
        vgg_probs = sess.run(vgg_prob,  feed_dict={X: X_batch, y: Y_batch} )
        inc_probs = sess.run(inc_prob,  feed_dict={X: X_batch, y: Y_batch} )
        res_probs = sess.run(res_prob,  feed_dict={X: X_batch, y: Y_batch} )
        
        tmp = np.zeros((batch_size,n_outputs))
        for i in range(batch_size):
            tmp[i,np.argmax(vgg_probs[i])] += 1
            tmp[i,np.argmax(inc_probs[i])] += 1
            tmp[i,np.argmax(res_probs[i])] += 1
        
        
        tmp= np.argmax(tmp,axis=1)
        
        res.append(tmp)
    return np.concatenate(res)

def avg(X_all,Y_all,batch_size):
    
    res = []
    for i in range(len(X_all)//batch_size):
        X_batch = X_all[i*batch_size:(i+1)*batch_size, :,:,:]
        Y_batch = Y_all[i*batch_size:(i+1)*batch_size]
        vgg_probs = sess.run(vgg_prob,  feed_dict={X: X_batch, y: Y_batch} )
        inc_probs = sess.run(inc_prob,  feed_dict={X: X_batch, y: Y_batch} )
        res_probs = sess.run(res_prob,  feed_dict={X: X_batch, y: Y_batch} )
        
        tmp = np.zeros((batch_size,n_outputs))
        for i in range(batch_size):
            three = np.vstack([vgg_probs[i], inc_probs[i], res_probs[i]])
            tmp[i] = np.mean(three, axis=0)

        tmp= np.argmax(tmp,axis=1)
        
        res.append(tmp)
    return np.concatenate(res)

    
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



best_model_params = None 

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

#with tf.Session() as sess:
#    init.run()
#    
##    saver_all.restore(sess, './my_all')
##    saver_pool.restore(sess, './pool/my_pool')
##    saver_vgg.restore(sess, './vgg_v2/my_vgg')
##    saver_inc.restore(sess, './inc_v2/my_inc')
##    saver_res.restore(sess, './res_v2/my_res')
#    
#    start_time = timeit.default_timer()
#    Y_vali = validation_raw['emotion'].tolist()
#    X_vali = process_many_parallel(validation_raw['pixels'].tolist())
#    
#    for epoch in range(n_epochs):
#        this_time = timeit.default_timer()
#        for iteration in range(int(len(data) * train_ratio) // batch_size):
#            train_batch_raw = training_raw.sample(n=batch_size, replace=True)
#            Y_batch = train_batch_raw['emotion'].tolist()
#            X_batch = process_many_parallel(train_batch_raw['pixels'].tolist())
#            sess.run(pool_training_op, feed_dict={X: X_batch, y: Y_batch, training: True})
#            if iteration % check_interval == 0:
#                
#                loss_val = pool_eval_loss_in_batches(X_vali, Y_vali, batch_size)
##                acc_val = vgg_eval_accuracy_in_batches(X_vali,Y_vali, batch_size)
##                print('Current validation accuracy: {:.4f}%'.format(acc_val*100))
#                if loss_val < best_loss_val:
#                    best_loss_val = loss_val
#                    checks_since_last_progress = 0
#                    best_model_params = get_model_params()
#                else:
#                    checks_since_last_progress += 1
#                    
##                step = epoch * int(len(data) * train_ratio) // batch_size + iteration
##                summary_str = vgg_accuracy_summary.eval(feed_dict={X: X_batch, y: Y_batch})
##                file_writer.add_summary(summary_str, step)
#        acc_train = pool_accuracy.eval(feed_dict={X: X_batch, y: Y_batch})
#        acc_val = pool_eval_accuracy_in_batches(X_vali,Y_vali, batch_size)
#        print("Epoch {}, train accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}, time per epoch: {:.4f}".format(
#                  epoch, acc_train * 100, acc_val * 100, best_loss_val, timeit.default_timer()-this_time))
#        if checks_since_last_progress > max_checks_without_progress:
#            print("Early stopping!")
#            break
#
#    if best_model_params:
#        restore_model_params(best_model_params)
#        
#    Y_test = test_raw['emotion'].tolist()
#    X_test = process_many_parallel(test_raw['pixels'].tolist())
#    acc_test = pool_eval_accuracy_in_batches(X_test,Y_test, batch_size)
#    print("Final accuracy on test set: {:.4f}%, total time: {:.2f}".format(acc_test*100,
#          timeit.default_timer()- start_time))
##    saver_vgg.save(sess, "./my_vgg")
##    saver_inc.save(sess, "./my_inc")
##    saver_res.save(sess, "./my_res")
##    saver_pool.save(sess, "./my_pool")
##    saver_all.save(sess, "./my_all")
#    
#    
#
#file_writer.flush()
#file_writer.close()


#with tf.Session() as sess:
#    init.run()
#    
##    saver_all.restore(sess, './my_all')
#    
#    saver_vgg.restore(sess, './vgg_v2/my_vgg')
#    saver_inc.restore(sess, './inc_v2/my_inc')
#    saver_res.restore(sess, './res_v2/my_res')
#    saver_pool.restore(sess, './pool/my_pool')
#    
#    start_time = timeit.default_timer()
#    Y_vali = validation_raw['emotion'].tolist()
#    X_vali = process_many_parallel(validation_raw['pixels'].tolist())
#    
#
##    preds_vali = voting(X_vali,Y_vali, batch_size)
##    correct_vali = np.equal(preds_vali, Y_vali[:len(preds_vali)])
##    acc_vali = np.mean(correct_vali)
#    acc_vali = pool_eval_accuracy_in_batches(X_vali,Y_vali, batch_size)
#    print("Accuracy on validation set: {:.4f}%, total time: {:.2f}".format(acc_vali*100,
#          timeit.default_timer()- start_time))
#
#        
#    Y_test = test_raw['emotion'].tolist()
#    X_test = process_many_parallel(test_raw['pixels'].tolist())
##    preds_test = voting(X_test,Y_test, batch_size)
##    correct_test = np.equal(preds_test, Y_test[:len(preds_test)])
##    acc_test = np.mean(correct_test)
#    acc_test = pool_eval_accuracy_in_batches(X_test,Y_test, batch_size)
#    print("Accuracy on test set: {:.4f}%, total time: {:.2f}".format(acc_test*100,
#          timeit.default_timer()- start_time))
#    
#    test_pred = pool_pred_in_batches(X_test,Y_test, batch_size)
#    confusion = tf.confusion_matrix(Y_test[:test_pred.shape[0]],test_pred).eval()
#    
#    confusion = confusion / int(test_pred.shape[0])*100
#    df_cm = pd.DataFrame(confusion, index = [Y_dic[i] for i in Y_dic.keys()],
#                  columns = [Y_dic[i] for i in Y_dic.keys()])
#    plt.figure(figsize = (10,7))
#    sn.heatmap(df_cm, annot=True)


with tf.Session() as sess:
    init.run()
    
#    saver_all.restore(sess, './my_all')
    
    saver_vgg.restore(sess, './vgg_v2/my_vgg')
    saver_inc.restore(sess, './inc_v2/my_inc')
    saver_res.restore(sess, './res_v2/my_res')
    saver_pool.restore(sess, './pool/my_pool')
    
    start_time = timeit.default_timer()
    Y_vali = validation_raw['emotion'].tolist()
    X_vali = process_many_parallel(validation_raw['pixels'].tolist())
    

    preds_vali = avg(X_vali,Y_vali, batch_size)
    correct_vali = np.equal(preds_vali, Y_vali[:len(preds_vali)])
    acc_vali = np.mean(correct_vali)
#    acc_vali = pool_eval_accuracy_in_batches(X_vali,Y_vali, batch_size)
    print("Accuracy on validation set: {:.4f}%, total time: {:.2f}".format(acc_vali*100,
          timeit.default_timer()- start_time))

        
    Y_test = test_raw['emotion'].tolist()
    X_test = process_many_parallel(test_raw['pixels'].tolist())
    preds_test = avg(X_test,Y_test, batch_size)
    correct_test = np.equal(preds_test, Y_test[:len(preds_test)])
    acc_test = np.mean(correct_test)
#    acc_test = pool_eval_accuracy_in_batches(X_test,Y_test, batch_size)
    print("Accuracy on test set: {:.4f}%, total time: {:.2f}".format(acc_test*100,
          timeit.default_timer()- start_time))
    
#    test_pred = pool_pred_in_batches(X_test,Y_test, batch_size)
#    confusion = tf.confusion_matrix(Y_test[:test_pred.shape[0]],test_pred).eval()
#    
#    confusion = confusion / int(test_pred.shape[0])*100
#    df_cm = pd.DataFrame(confusion, index = [Y_dic[i] for i in Y_dic.keys()],
#                  columns = [Y_dic[i] for i in Y_dic.keys()])
#    plt.figure(figsize = (10,7))
#    sn.heatmap(df_cm, annot=True)