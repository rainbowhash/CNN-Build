# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:44:45 2021

@author: Tharun
"""
import numpy as np

class softmax:
    def __init__(self,size,nodes):
        self.weights=np.random.rand(size,nodes)/size
        # each pixel will have a weight hence width*hight*channel
        # dividing by size reduce the variance of the weights.
        self.bias=np.zeros(nodes)
        
        
    def forward(self,image):
        
        image=image.flatten()
        # the network needs it flatten and not in its shape
        # dot operation of a matrix is smilar to what happens after flattening 
        # flattening involves converting rows as colum and concatinati it with
        # new row 
        # equation of softmax is e^of class/sum(e^(all class))
        sum=np.dot(image,self.weights)+self.bias#
        print(sum)
        exp=np.exp(sum)
        denominator=np.sum(exp,axis=0)
        return exp/denominator
        
        
        