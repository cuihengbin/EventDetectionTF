'''
---------------------------------------------------------------------------------
SUMMARY:  preprocess of wav data
AUTHOR:   Qingkai WEI, revised from DCASE2016 Qiuqiang KONG
Created:  2016.12.28
Modified:
---------------------------------------------------------------------------------
'''
import numpy as np
import cPickle
import os
import sys
import matplotlib.pyplot as plt
import wavio
import librosa
import config as cfg
import csv
import scipy.stats
import tensorflow as tf
from scipy import signal

### readwav
def readwav( path ):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs

# calculate mel feature
def GetMel( wav_fd, fe_fd, n_delete ):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.16kHz.wav') ]
    names = sorted(names)
    #print len(names)
    for na in names:
        print na
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )
        assert fs==cfg.fs
        ham_win = np.hamming(cfg.win)
        [f, t, X] = signal.spectral.spectrogram( wav, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
        X = X.T
        
        # define global melW, avoid init melW every time, to speed up. 
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel( fs, n_fft=cfg.win, n_mels=40, fmin=0., fmax=8000 )
            melW /= np.max(melW, axis=-1)[:,None]
            
        X = np.dot( X, melW.T )
        X = X[:, n_delete:]
        
        # DEBUG. print mel-spectrogram
        #plt.matshow(np.log(X.T), origin='lower', aspect='auto')
        #plt.show()
        #pause

        out_path = fe_fd + '/' + na[0:-10] + '.f'
        cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )

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


### format label
# get tags
def GetTags( info_path ):
    with open( info_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    tags = lis[-2][1]
    return tags
            
# tags to categorical, shape: (n_labels)
def TagsToCategory( tags ):
    y = np.zeros( len(cfg.labels) )
    for ch in tags:
        y[ cfg.lb_to_id[ch] ] = 1
    return y

### if set fold=None means use all data as training data
def GetAllData( fe_fd, agg_num, hop, fold ):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
    tr_na_list, te_na_list = [], []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        info_path = cfg.dev_wav_fd + '/' + na + '.csv'
        tags = GetTags( info_path )
        y = TagsToCategory( tags )
        X = cPickle.load( open( fe_path, 'rb' ) )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )

        if curr_fold==fold:
            te_Xlist.append( X3d )
            te_ylist += [ y ] * len( X3d )
            te_na_list.append( na )
        else:
            tr_Xlist.append( X3d )
            tr_ylist += [ y ] * len( X3d )
            tr_na_list.append( na )

    if fold is None:
        return np.concatenate( tr_Xlist, axis=0 ), np.array( tr_ylist ), tr_na_list
    else:
        return np.concatenate( tr_Xlist, axis=0 ), np.array( tr_ylist ), tr_na_list, \
           np.concatenate( te_Xlist, axis=0 ), np.array( te_ylist ), te_na_list
    
###
# create an empty folder
def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)

def mat_3d_to_2d(X):
    # [batch_num, n_time, n_freq] --> [batch_num, n_time * n_freq]
    [N, n_row, n_col] = X.shape
    return X.reshape( (N, n_row*n_col) )


if __name__ == "__main__":
    agg_num = 11  # concatenate frames
    hop = 1  # step_len
    fold = 0

    CreateFolder( cfg.dev_fe_fd )
    CreateFolder( cfg.dev_fe_mel_fd )
    GetMel( cfg.dev_wav_fd, cfg.dev_fe_mel_fd, n_delete=0 )

    tr_X, tr_y, _, te_X, te_y, te_na_list = GetAllData(cfg.dev_fe_mel_fd, agg_num, hop, fold)
    [batch_num, n_time, n_freq] = tr_X.shape
    tr_X = mat_3d_to_2d(tr_X)
    te_X = mat_3d_to_2d(te_X)

    writer = tf.python_io.TFRecordWriter('train.tfrecords')
    for i in xrange(len(tr_X)):
        example = tf.train.Example(features=tf.train.Features(feature={
            'x_train': tf.train.Feature(float_list=tf.train.FloatList(value=tr_X[i,:] )),
            'y_train': tf.train.Feature(float_list=tf.train.FloatList(value=tr_y[i,:] ))
        }))
        writer.write(example.SerializeToString())
    writer.close()

    writer = tf.python_io.TFRecordWriter('test.tfrecords')
    for i in xrange(len(te_X)):
        example = tf.train.Example(features=tf.train.Features(feature={
            'x_train': tf.train.Feature(float_list=tf.train.FloatList(value=te_X[i,:] )),
            'y_train': tf.train.Feature(float_list=tf.train.FloatList(value=te_y[i,:] ))
        }))
        writer.write(example.SerializeToString())
    writer.close()
