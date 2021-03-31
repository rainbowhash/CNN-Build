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




class cnn:
    def __init__(self):
        self.conv=conv3x3(8)
        # convolution helps in capturing patterns in image.
         # a solbel filter can capture the edge in a image,smillary convolution will 
         # be capturing pattern in images
        self.mp=maxpool(2)
         # max pooling is layer where the image is reduced in dimention but mainly
         # in a image after convoling there is redudent pattrn in the nighbouring pixel
         # example when we find the edges in a image and consider a pixel its 
         # nighbouring pixel will contain just the edge and no other new information.
         # Hence by maxpooling we capture non redudent and unique patterns in the image
        self.sb=softmax(13*13*8,10) 
         # a softmax activation function generate a porbality distrubution for our prediction
         # using a softmax in our architecture node generates a probablity of a image to
         # belong to that node or that node that represents the class
         # another advantage is to use cross entrop loss for backpropogation of the network
        self.prep_data()
        
    def forwardpro(self,image,label):
        layer1=self.conv.forward_pass(image)
        layer2=self.mp.forward(layer1)
        layer3=self.sb.forward(layer2)
        loss=-np.log(layer3[label])
        acc= 1 if np.argmax(layer3)==label else 0
        return layer3,loss,acc
    
    def train(self,image,label,lr=.005):
        gradiant=np.zeros(10)
        # we need to calculate the gradiant of all the nodes
        z,lo,ac=self.forwardpro(image,label)
        gradiant[label]=-1/z[label]
        gradiant=self.sb.backprop(gradiant,lr)
        #print(lo,ac)
        return lo,ac

    def genisis(self):
        loss = 0
        num_correct = 0
        for i, (im, label) in enumerate(zip(self.x_train, self.y_train)):
            if i % 100 == 99:
                print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i + 1, loss / 100, num_correct))
                loss = 0
                num_correct = 0
            l, acc = self.train(im, label)
            loss += l
            num_correct += acc
        
    def prep_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

        
    
if __name__=="__main__":
    cnn=cnn()
    cnn.genisis()
    
    
    
    
