'''
---------------------------------------------------------------------------------
SUMMARY:  train and evaluate the precision, recall and F-value on cross fold set
AUTHOR:   Qingkai WEI, revised from DCASE2016 Qiuqiang KONG
Created:  2016.12.28
Modified:
---------------------------------------------------------------------------------
'''
import pickle
import numpy as np
import tensorflow as tf
import config as cfg
import prepare_data as pp_dev_data
import csv
import cPickle
np.random.seed(1515)

# hyper-params
agg_num = 11       # concatenate frames
hop = 1            # step_len
act = 'relu'
n_hid = 500
n_out = len( cfg.labels )
fold = 0

pos_weight = 3
starter_learning_rate = 0.0001
global_step = tf.Variable(0, trainable=False)
decay_rate = 0.8
decay_steps = 100
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate, staircase=True)

batch_size = 400
n_epochs = 500  # 21276
display_step = 10

n_input = 440
n_classes = len(cfg.labels)
dropout = 0.6

checkpoint_steps = 10
checkpoint_dir = 'md/'


def mat_3d_to_2d(X):
    # [batch_num, n_time, n_freq] --> [batch_num, n_time * n_freq]
    [N, n_row, n_col] = X.shape
    return X.reshape( (N, n_row*n_col) )

# prepare data
# tr_X, tr_y, _ = pp_dev_data.GetAllData( cfg.dev_fe_mel_fd, agg_num, hop, fold=None )
tr_X, tr_y, _, te_X, te_y, te_na_list = pp_dev_data.GetAllData(cfg.dev_fe_mel_fd, agg_num, hop, fold)
[batch_num, n_time, n_freq] = tr_X.shape

tr_X = mat_3d_to_2d(tr_X)
te_X = mat_3d_to_2d(te_X)


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

def dnn(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.nn.dropout(_X, _dropout)
    d1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(_X, _weights['wd1']), _biases['bd1']), name="d1")

    d2x = tf.nn.dropout(d1, _dropout)
    d2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(d2x, _weights['wd2']), _biases['bd2']), name="d2")

    d3x = tf.nn.dropout(d2, _dropout)
    d3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(d3x, _weights['wd3']), _biases['bd3']), name="d3")

    dout = tf.nn.dropout(d3, _dropout)
    # Output, class prediction
    out = tf.nn.sigmoid(tf.matmul(dout, _weights['out']) + _biases['out'])
    return out
# Store layers weight & bias
weights = {
    'wd1': tf.Variable(tf.random_normal([440, n_hid], stddev=0.01)),
    'wd2': tf.Variable(tf.random_normal([n_hid, n_hid], stddev=0.01)),
    'wd3': tf.Variable(tf.random_normal([n_hid, n_hid], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_hid, 8]))
}
biases = {
    'bd1': tf.Variable(tf.random_normal([n_hid])),
    'bd2': tf.Variable(tf.random_normal([n_hid])),
    'bd3': tf.Variable(tf.random_normal([n_hid])),
    'out': tf.Variable(tf.random_normal([8])),
}
# Construct model
pred = dnn(x, weights, biases, keep_prob)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(pred, y, pos_weight, name=None))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing the variables
init = tf.initialize_all_variables()
'''
saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Fit training using batch data
    batch_n = int(batch_num/batch_size)
    i = 0
    while i < n_epochs:  #range(batch_num):
        for i2 in xrange(batch_n):
            if (i2) * batch_size < batch_num:
                itemp = i2
                batch_x = tr_X[(itemp) * batch_size: min((itemp + 1) * batch_size, batch_num), :]
                batch_y = tr_y[(itemp) * batch_size: min((itemp + 1) * batch_size, batch_num), :]
            else:
                itemp = i2 % int(batch_num / batch_size)
                batch_x = tr_X[(itemp) * batch_size: min((itemp + 1) * batch_size, batch_num), :]
                batch_y = tr_y[(itemp) * batch_size: min((itemp + 1) * batch_size, batch_num), :]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            ya = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            # Calculate batch accuracy
        if i % display_step == 0:
            print "Testing Accuracy and Loss:", i,\
                (sess.run(accuracy, feed_dict={x: tr_X, y: tr_y, keep_prob: 1.})),\
                (sess.run(cost, feed_dict={x: tr_X, y: tr_y, keep_prob: 1.})),\
                (sess.run(accuracy, feed_dict={x: te_X, y: te_y, keep_prob: 1.})),\
                (sess.run(cost, feed_dict={x: te_X, y: te_y, keep_prob: 1.}))
        if i % checkpoint_steps == 0:
            saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i)
        i = i + 1
'''


# to get Prec, Recall, Fvalue
thres = 0.4  # thres, tune to prec=recall
n_labels = len(cfg.labels)
fe_fd = cfg.dev_fe_mel_fd
# concatenate feautres
def mat_2d_to_3d(X, agg_num, hop):
# pad to at least one block
    len_X, n_in = X.shape
    if (len_X < agg_num):
        X = np.concatenate((X, np.zeros((agg_num - len_X, n_in))))
    # agg 2d to 3d
    len_X = len(X)
    i1 = 0
    X3d = []
    while (i1 + agg_num <= len_X):
        X3d.append(X[i1:i1 + agg_num])
        i1 += hop
    return np.array(X3d)

def tp_tn_fp_fn( p_y_pred, y_gt, thres ):
    y_pred = np.zeros_like( p_y_pred )
    y_pred[ np.where( p_y_pred>thres ) ] = 1.
    tp = np.sum( y_pred + y_gt > 1.5 )
    tn = np.sum( y_pred + y_gt < 0.5 )
    fp = np.sum( y_pred - y_gt > 0.5 )
    fn = np.sum( y_gt - y_pred > 0.5 )
    return tp, tn, fp, fn

def prec_recall_fvalue( p_y_pred, y_gt, thres ):
    tp, tn, fp, fn = tp_tn_fp_fn( p_y_pred, y_gt, thres )
    prec = tp / float( tp + fp )
    recall = tp / float( tp + fn )
    fvalue = 2 * ( prec * recall ) / ( prec + recall )
    return prec, recall, fvalue

gt_roll = []
pred_roll = []
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        pass

    with open(cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        # read one line
        for li in lis:
            na = li[1]
            curr_fold = int(li[2])
            if fold == curr_fold:
                # get features, tags
                fe_path = fe_fd + '/' + na + '.f'
                info_path = cfg.dev_wav_fd + '/' + na + '.csv'
                tags = pp_dev_data.GetTags(info_path)
                y = pp_dev_data.TagsToCategory(tags)
                X = cPickle.load(open(fe_path, 'rb'))

                # aggregate data
                X3d = mat_2d_to_3d(X, agg_num, hop)
                te_xsingle = mat_3d_to_2d(X3d)

                p_y_pred = sess.run(pred, feed_dict={x: te_xsingle, keep_prob: 1.0})
                p_y_tempb = np.mean(p_y_pred, axis=0)  # shape:(n_label)
                pred_out = np.zeros(n_labels)
                pred_out[np.where(p_y_tempb > thres)] = 1
                pred_roll.append(pred_out)
                gt_roll.append(y)

pred_roll = np.array(pred_roll)
gt_roll = np.array(gt_roll)
# calculate prec, recall, fvalue
prec, recall, fvalue = prec_recall_fvalue(pred_roll, gt_roll, thres)
print prec, recall, fvalue
