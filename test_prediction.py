#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Total params: 15,857,665
#Trainable params: 12,386,689
#Non-trainable params: 3,470,976

import numpy as np
import os, glob
from keras.preprocessing import image as kImage
from skimage.transform import pyramid_gaussian
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
    s2 = []
    s3 = []
    for x in s1:
       pyramid = tuple(pyramid_gaussian(x/255., max_layer=2, downscale=2))
       s2.append(pyramid[1]*255.)
       s3.append(pyramid[2]*255.)
    s2 = np.asarray(s2)
    s3 = np.asarray(s3)
    
    return [s1, s2, s3]


# In[2]:


# get data
#input_path = glob.glob(os.path.join('sample_test_frames', 'highway', '*.jpg')) # path to your test frames
input_path = glob.glob(os.path.join('sample_test_frames', 'highway', '*.png')) # path to your test frames
data = getData(input_path)
print (data[0].shape, data[1].shape, data[2].shape)


# In[3]:


# plot last frame in 3 diff scales
num_in_row = 1
num_in_col = 3
frame_idx = 3 # display frame index

plt.rcParams['figure.figsize'] = (10.0, 7.0) # set figure size

for i in range(num_in_row * num_in_col):
    x = data[i][frame_idx]
    plt.subplot(num_in_row, num_in_col, i+1)
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

# a sample FgSegNet_M model can be downloaded at https://drive.google.com/open?id=1KiEChAxuweEZHwqT5HrkTCbSjT9MxwUF
#mdl_path = 'mdl_highway.h5'
mdl_path =  './FgSegNet_M/CDnet/models50/baseline/mdl_office.h5'

#model = load_model(mdl_path, custom_objects={'MyUpSampling2D': MyUpSampling2D}) # load the saved model that is trained with 50 frames
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




