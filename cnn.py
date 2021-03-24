# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:11:08 2021

@author: Tharun 
"""

from conv import conv3x3
from maxpool import maxpool
from softmax import softmax
import tensorflow as tf
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


conv=conv3x3(8)
 # convolution helps in capturing patterns in image. 
 # a solbel filter can capture the edge in a image,smillary convolution will 
 # be capturing pattern in images



mp=maxpool(2)
# max pooling is layer where the image is reduced in dimention but mainly
# in a image after convoling there is redudent pattrn in the nighbouring pixel
# example when we find the edges in a image and consider a pixel its 
# nighbouring pixel will contain just the edge and no other new information.
# Hence by maxpooling we capture non redudent and unique patterns in the image

sb=softmax(13*13*8,10) 
# a softmax activation function generate a porbality distrubution for our prediction
# using a softmax in our architecture node generates a probablity of a image to
# belong to that node or that node that represents the class
# another advantage is to use cross entrop loss for backpropogation of the network

def forwardpro(image,label):
    layer1=conv.forward_pass(image)
    layer2=mp.forward(layer1)
    layer3=sb.forward(layer2)
    loss=-np.log(layer3[label])
    acc= 1 if np.argmax(layer3)==label else 0
    return layer3,loss,acc

loss=0
acc=0
for i, (image,labe) in enumerate(zip(x_train[:10],y_train[:10])):
    layer,lo,ac=forwardpro(image,labe)
    loss+=lo
    acc+=ac
    print(acc)
    
    
    
