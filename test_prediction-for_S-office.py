#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 12:07:57 2018

@author: rajs
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Total params: 15,857,665
#Trainable params: 12,386,689
#Non-trainable params: 3,470,976

import numpy as np
import os, glob
from keras.preprocessing import image as kImage
from keras.models import load_model
import matplotlib.pyplot as plt
from FgSegNet.my_upsampling_2d import MyUpSampling2D

def getData(input_path):
    X = []
    for p in input_path:
        x = kImage.load_img(p)
        x = kImage.img_to_array(x)
        X.append(x)
    X = np.asarray(X)
    
    s1 = X
    del X
    
    return s1


# In[2]:


# get data
input_path = glob.glob(os.path.join('sample_test_frames', 'office', '*.jpg')) # path to your test frames
data = getData(input_path)

print (data.shape)


# In[3]:


# plot last frame in 3 diff scales
frame_idx = 0 # display frame index

plt.rcParams['figure.figsize'] = (10.0, 7.0) # set figure size


x = data[frame_idx]
plt.subplot(1, 1, 1)
plt.imshow(x.astype('uint8'))

plt.title(x.shape)
plt.axis('off')
    
plt.show()

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

# In[4]:
# Segment on a single CPU for this test
#mdl_path = 'mdl_highway.h5'
#mdl_path =  './FgSegNet_S/CDnet/models50/baseline/mdl_highway.h5'
mdl_path =  './FgSegNet_S/CDnet/models50/baseline/mdl_office.h5'

model = load_model(mdl_path, custom_objects={'MyUpSampling2D': MyUpSampling2D,
                                              'loss' : loss}) 
                                            
probs = model.predict(data, batch_size=1, verbose=1)
print(probs.shape) # (5, 240,320,1)
probs = probs.reshape([probs.shape[0], probs.shape[1], probs.shape[2]])
print(probs.shape) # (5, 240,320)


# In[6]:


# plot the first segmentation mask
x = probs[frame_idx]

plt.subplot(1, 1, 1)
plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams['image.cmap'] = 'gray'

plt.imshow(x)

plt.title('seg. mask of shape ' + str(x.shape))
plt.axis('off')
plt.show()


# In[7]:


# Thresholding (one can specify any threshold values)
threshold = 0.8
x[x<threshold] = 0.
x[x>=threshold] = 1.

plt.subplot(1, 1, 1)
plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams['image.cmap'] = 'gray'

plt.imshow(x)

plt.title('seg. mask of shape ' + str(x.shape))
plt.axis('off')
plt.show()