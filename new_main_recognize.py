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

global_step = tf.Variable(0, trainable=False)

n_input = 440
n_classes = len(cfg.labels)
dropout = 0.6

checkpoint_steps = 10
checkpoint_dir = 'checkpoint_dir/'


def mat_3d_to_2d(X):
    # [batch_num, n_time, n_freq] --> [batch_num, n_time * n_freq]
    [N, n_row, n_col] = X.shape
    return X.reshape( (N, n_row*n_col) )
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
# Read the data to be recognized
def GetAllEvaData(fe_fd, csv_file, agg_num, hop):
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    Xlist = []
    # read one line
    for li in lis:
        na = li[1]

        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        X = cPickle.load(open(fe_path, 'rb'))

        # aggregate data
        X3d = mat_2d_to_3d(X, agg_num, hop)
        Xlist.append(X3d)
    return np.concatenate(Xlist, axis=0)


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

# Initializing the variables
init = tf.initialize_all_variables()


# prepare data
te_X = GetAllEvaData( cfg.dev_fe_mel_fd, cfg.eva_csv_path, agg_num, hop )
# do recognize and evaluation
thres = 0.4  # thres, tune to prec=recall
pp_dev_data.CreateFolder(cfg.eva_results_fd)
txt_out_path = cfg.eva_results_fd + '/task4_results.txt'
fwrite = open(txt_out_path, 'w')

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        pass

    with open(cfg.eva_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        # read one line
        for li in lis:
            na = li[1]
            full_na = na + '.16kHz.wav'
            # get features, tags
            fe_path = cfg.dev_fe_mel_fd + '/' + na + '.f'
            X = cPickle.load(open(fe_path, 'rb'))

            # aggregate data
            X3d = mat_2d_to_3d(X, agg_num, hop)
            te_xsingle = mat_3d_to_2d(X3d)
            p_y_pred = sess.run([pred], feed_dict={x: te_xsingle, keep_prob: 1.0})

            test = p_y_pred[0]
            p_y_pred = np.mean(test, axis=0)  # shape:(n_label)
            #  write out data
            for j1 in xrange(7):
                fwrite.write(full_na + ',' + cfg.id_to_lb[j1] + ',' + str(p_y_pred[j1]) + '\n')
fwrite.close()
print "Write out to", txt_out_path, "successfully!"
