#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 12:10:43 2018

@author: rajs
"""

import cv2
print(cv2.__version__)

import numpy as np
from keras.preprocessing import image as kImage
from skimage.transform import pyramid_gaussian
from keras.models import load_model
import matplotlib.pyplot as plt
from FgSegNet.my_upsampling_2d import MyUpSampling2D

#import h5py  
#
#mdl_path = 'mdl_office.h5'
mdl_path =  './FgSegNet_S/CDnet/models50/baseline/mdl_office.h5'

import keras.backend as K
import tensorflow as tf
def loss(y_true, y_pred):
    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


model = load_model(mdl_path, custom_objects={'MyUpSampling2D': MyUpSampling2D,
                                              'loss' : loss}) 
print(model.summary()) 

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot_MyTrain.png', show_shapes=True, show_layer_names=True)
# load the saved model that is trained with 50 frames

        
from PIL import Image

video_capture = cv2.VideoCapture(0)       
while cv2.waitKey(10) < 0:
    ret, frame = video_capture.read()           
    if ret:               
        
        cv2.imshow('Input',frame)                     
        frame = cv2.resize(frame, (360, 240))        
        frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)        
        pil_im = Image.fromarray(frame2)
        
#        pil_im.show()

        s1 = kImage.img_to_array(pil_im)    
        s1 = np.expand_dims(s1, axis=0)

        data = s1
        print (data.shape)
        
        frame_idx = 0 # display frame index
        
        probs = model.predict(data, batch_size=1, verbose=1)
        print(probs.shape) 
        probs = probs.reshape([probs.shape[0], probs.shape[1], probs.shape[2]])
        print(probs.shape) 
        
        # plot the first segmentation mask
        x = probs[frame_idx]
        threshold = 0.8
        x[x<threshold] = 0
        x[x>=threshold] = 255
        
        cv2.imshow('Foreground',x)
        cv2.waitKey(10)
    else:
        print('No Frame')
        continue
    
video_capture.release()
cv2.destroyAllWindows()
print('Released')
plt.close()
