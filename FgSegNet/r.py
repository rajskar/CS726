#@title Default title text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 22:22:22 2017

@author: longang
"""

import numpy as np
import tensorflow as tf
import random as rn
import os, sys

# set current working directory
cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)

# =============================================================================
#  For reprodocable results
# =============================================================================
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads = 1, 
                              inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

K.set_session(sess)

import keras, glob
from keras.preprocessing import image as kImage
from skimage.transform import pyramid_gaussian
from sklearn.utils import compute_class_weight
from FgSegNet_M_S_module_3D import FgSegNet_M_S_module
from keras.utils.data_utils import get_file



def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

# alert the user
if keras.__version__!= '2.0.6' or tf.__version__!='1.1.0' or sys.version_info[0]<3:
  print('We implemented using [keras v2.0.6, tensorflow-gpu v1.1.0, python v3.6.3], other versions than these may cause errors somehow!\n')

# =============================================================================
# Few frames, load into memory directly
# =============================================================================
def generateData(train_dir, dataset_dir, scene, method_name, seq_len = 5):
    assert method_name in ['FgSegNet_M', 'FgSegNet_S'], 'method_name is incorrect'
    
    void_label = -1.
    
    # Given ground-truths, load training frames
    # ground-truths end with '*.png'
    # training frames end with '*.jpg'
    
    # given ground-truths, load inputs      
#    Y_list = sorted(glob.glob(os.path.join(train_dir, '*.png')))
#    X_list = sorted(glob.glob(os.path.join(dataset_dir, 'input','*.jpg')))

    print(dataset_dir)    
    Y_list = sorted(glob.glob(os.path.join(dataset_dir, 'groundtruth', '*.png')))
    X_list = sorted(glob.glob(os.path.join(dataset_dir, 'input','*.jpg')))
    
    print(len(X_list))
    print(len(Y_list))
    
    nX_list = []
    for i in range(len(X_list)):
        j = X_list[i].find('(')
        if j == -1:
            nX_list.append(X_list[i])
    
    X_list = nX_list

    nY_list = []
    for i in range(len(Y_list)):
        j = Y_list[i].find('(')
        if j == -1:
            nY_list.append(Y_list[i])
    
    Y_list = nY_list
    print(len(X_list))
    print(len(Y_list))    
    
    if len(Y_list)<=0 or len(X_list)<=0:
        raise ValueError('System cannot find the dataset path or ground-truth path. Please give the correct path.')
        
    
    if len(X_list)!=len(Y_list):
        raise ValueError('The number of X_list and Y_list must be equal.')
        
    # load training data
    X = []
    Y = []
    for i in range(len(X_list)):
        x = kImage.load_img(X_list[i])
        x = kImage.img_to_array(x)
        X.append(x)
        
        x = kImage.load_img(Y_list[i], grayscale = True)
        x = kImage.img_to_array(x)
        shape = x.shape
        x /= 255.0
        x = x.reshape(-1)
        idx = np.where(np.logical_and(x>0.25, x<0.8))[0] # find non-ROI
        if (len(idx)>0):
            x[idx] = void_label
        x = x.reshape(shape)
        x = np.floor(x)
        Y.append(x)
        
    X = np.asarray(X)
    Y = np.asarray(Y)
        
    # We do not consider temporal data
    idx = list(range(X.shape[0]))
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    
    if method_name=='FgSegNet_M':
        # Image Pyramid
        scale2 = []
        scale3 = []
        for i in range(0, X.shape[0]):
          pyramid = tuple(pyramid_gaussian(X[i]/255., max_layer=2, downscale=2))
          scale2.append(pyramid[1]*255.) # 2nd scale
          scale3.append(pyramid[2]*255.) # 3rd scale
          del pyramid
           
        scale2 = np.asarray(scale2)
        scale3 = np.asarray(scale3)

    # compute class weights
    cls_weight_list = []
    for i in range(Y.shape[0]):
        y = Y[i].reshape(-1)
        idx = np.where(y!=void_label)[0]
        if(len(idx)>0):
            y = y[idx]
        lb = np.unique(y) #  0., 1
        cls_weight = compute_class_weight('balanced', lb , y)
        class_0 = cls_weight[0]
        class_1 = cls_weight[1] if len(lb)>1 else 1.0
                
        cls_weight_dict = {0:class_0, 1: class_1}
        cls_weight_list.append(cls_weight_dict)
#
    cls_weight_list = np.asarray(cls_weight_list)
    
    Xseq = []
    scale2seq = []
    scale3seq = []
    Yseq = []
    cls_weight_listseq = []
    
    seq_len = 5
    for i in range(0, Y.shape[0], seq_len):
        Xseq.append(X[i : i + seq_len])
        scale2seq.append(scale2[i : i + seq_len])
        scale3seq.append(scale3[i : i + seq_len])
        Yseq.append(Y[i : i + seq_len])
        cls_weight_listseq.append(cls_weight_list[i : i + seq_len])
    del X, scale2, scale3, Y, cls_weight_list        
    
    Xseq      = np.array(Xseq)
    scale2seq = np.array(scale2seq)
    scale3seq = np.array(scale3seq)
    Yseq      = np.array(Yseq)
    cls_weight_listseq = np.array(cls_weight_listseq)
        
    if method_name=='FgSegNet_M':
        return [Xseq, scale2seq, scale3seq, Yseq, cls_weight_listseq]
    else:
        return [Xseq,Yseq,cls_weight_listseq]
    
def train(results, scene, mdl_path, vgg_weights_path, method_name):
    assert method_name in ['FgSegNet_M', 'FgSegNet_S'], 'method_name is incorrect'

    img_shape = results[0][0].shape # (seq, height, width, channel)
    model = FgSegNet_M_S_module(lr, reg, img_shape, scene, vgg_weights_path)
    
    if method_name=='FgSegNet_M':
        model = model.initModel_M('CDnet')
    else:
        model = model.initModel_S('CDnet')
    
    # make sure that training input shape equals to model output
    input_shape = (img_shape[0], img_shape[1], img_shape[2])
    output_shape = (model.output._keras_shape[1], model.output._keras_shape[2], model.output._keras_shape[3])
    assert input_shape==output_shape, 'Given input shape:' + str(input_shape) + ', but your model outputs shape:' + str(output_shape) 

    chk = keras.callbacks.ModelCheckpoint(mdl_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    redu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduce_factor, patience=num_patience, verbose=1, mode='auto')
    early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')

    if method_name=='FgSegNet_M':
        model.fit([results[0], results[1], results[2]], results[3], validation_split=val_split, epochs=max_epochs, batch_size=batch_size, 
                           callbacks=[redu, chk], verbose=1, class_weight=results[4], shuffle = True)
    else:
        # maybe we can use early stopping instead for FgSegNet_S, and also set max epochs to 100 or 110
        model.fit(results[0], results[1], validation_split=val_split, epochs=max_epochs+50, batch_size=batch_size, 
              callbacks=[redu, early], verbose=1, class_weight=results[2], shuffle = True)
        model.save(mdl_path)
     
    

    del model, results, chk, redu, early


# =============================================================================
# Main func
# =============================================================================
dataset = {
            'baseline':['highway']
}

# =============================================================================

method_name = 'FgSegNet_M' # either <FgSegNet_M> or <FgSegNet_S>, default FgSegNet_M

num_frames = 50 # either 50 or 200 frames, default 50 frames

reduce_factor = 0.1
num_patience = 6
lr = 1e-4
reg = 5e-4
max_epochs = 60 if num_frames==50 else 50 # 50f->60epochs, 200f->50epochs
val_split = 0.2
batch_size = 5
# =============================================================================

# Example: (free to modify)

# FgSegNet/FgSegNet/FgSegNet_M_S_CDnet.py
# FgSegNet/FgSegNet/FgSegNet_M_S_SBI.py
# FgSegNet/FgSegNet/FgSegNet_M_S_UCSD.py
# FgSegNet/FgSegNet/FgSegNet_M_S_module.py

# FgSegNet/FgSegNet_dataset2014/...
# FgSegNet/CDnet2014_dataset/...


assert num_frames in [50,200], 'Incorrect number of frames'
main_dir = os.path.join('..', method_name)
main_mdl_dir = os.path.join(main_dir, 'CDnet', 'models' + str(num_frames))
vgg_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
if not os.path.exists(vgg_weights_path):
    # keras func
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    vgg_weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP, cache_subdir='models',
                                file_hash='6d6bbae143d832006294945121d1f1fc')

print('*** Current method >>> ' + method_name + '\n')

for category, scene_list in dataset.items():
    
    mdl_dir = os.path.join(main_mdl_dir, category)
    if not os.path.exists(mdl_dir):
        os.makedirs(mdl_dir)
        
    for scene in scene_list:
        print ('Training ->>> ' + category + ' / ' + scene + ' New model 4')
        
        # training frame path and dataset2014 path
        train_dir = os.path.join('..', 'FgSegNet_dataset2014', category, scene + str(num_frames))
        dataset_dir = os.path.join('..', 'CDnet2014_dataset', category, scene)
        results = generateData(train_dir, dataset_dir, scene, method_name)

        mdl_path = os.path.join(mdl_dir, 'mdl_' + scene + '.h5')
        train(results, scene, mdl_path, vgg_weights_path, method_name)
        del results



