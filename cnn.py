# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:11:08 2021

@author: k20104661
"""

from conv import conv3x3
from maxpool import maxpool
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()



 # convolution helps in capturing patterns in image. 
 # a solbel filter can capture the edge in a image,smillary convolution will 
 # be capturing pattern in images
conv=conv3x3(8)
out=conv.forward_pass(x_train[0])

# max pooling is layer where the image is reduced in dimention but mainly
# in a image after convoling there is redudent pattrn in the nighbouring pixel
# example when we find the edges in a image and consider a pixel its 
# nighbouring pixel will contain just the edge and no other new information.
# Hence by maxpooling we capture non redudent and unique patterns in the image

mp=maxpool(2)
out=mp.forward(out)
print(out.shape)