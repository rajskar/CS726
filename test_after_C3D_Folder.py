import cv2
print(cv2.__version__)

import numpy as np
from keras.preprocessing import image as kImage
from skimage.transform import pyramid_gaussian
from keras.models import load_model
import matplotlib.pyplot as plt
from FgSegNet.my_upsampling_2d import MyUpSampling2D

mdl_path = './FgSegNet_M/CDnet/models50/baseline/Working/mdl_highway.h5'

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
plot_model(model, to_file='model_plot_MyTrain.png', show_shapes=True, 
           show_layer_names=True)
# load the saved model that is trained with 50 frames
from PIL import Image
i = 1
ft = 1
while cv2.waitKey(10) < 0:    
    X_list = []
    for j in range(5):
#        im_path = './CDnet2014_dataset/baseline/PETS2006/input/in{:06d}.jpg'.format(i)        
        im_path = './CDnet2014_dataset/baseline/highway/input/in{:06d}.jpg'.format(i)        
        i+=15
        X_list.append(im_path)
            
    print(X_list)
    
    # load training data
    s1 = []
    f = []
    for k,l in enumerate(X_list):
        frame = cv2.imread(l)
        frame = cv2.resize(frame, (320, 240))        
        f.append(frame)
#        cv2.imshow('Input' + str(k),frame)        
        frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)        
        pil_im = Image.fromarray(frame2)        
        s1.append(kImage.img_to_array(pil_im))
    
    s1 = np.asarray(s1)
   
    
        
    scale2 = []
    scale3 = []
    for x in s1:
        print(x.shape)
        pyramid = tuple(pyramid_gaussian(x/255., max_layer=2, downscale=2))
        scale2.append(pyramid[1]*255.) # 2nd scale
        scale3.append(pyramid[2]*255.) # 3rd scale
        del pyramid
    
    s1 = np.expand_dims(s1, axis=0)
    s2 = np.asarray(scale2)
    
    s2 = np.expand_dims(s2, axis=0)
    s3 = np.asarray(scale3)
    s3 = np.expand_dims(s3, axis=0)
        
    data = [s1, s2, s3]
    
    probs = model.predict(data, batch_size=1, verbose=1)
    probs = probs.reshape([probs.shape[1], probs.shape[2], probs.shape[3]])
    
    for j in range(5):
        x = probs[j]
        threshold = 0.8
        x[x<threshold] = 0
        x[x>=threshold] = 1        
        cv2.imshow('Foreground'+str(j), x)
        x = x.astype(np.uint8) 
                
        _, contours, _ = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(f[j], (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Contours' +str(j), f[j])

    if ft == 1:
        print('Arrange Image windows and press any key to start')
        ft = 0
        cv2.waitKey(0)
            
cv2.destroyAllWindows()
print('Released')
plt.close()
