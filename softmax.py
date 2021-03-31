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
        self.last_input_shape=image.shape
        
        image=image.flatten()
        self.last_input=image
        # the network needs the image to be flatten and not in its orgi shape
        # dot operation of a matrix is smilar to what happens after flattening 
        # flattening involves converting rows as colum and concatinati it with
        # new row 
        # equation of softmax is e^of class/sum(e^(all class))
        sum=np.dot(image,self.weights)+self.bias
        #print(sum)
        self.last_sum=sum
        exp=np.exp(sum)
        denominator=np.sum(exp,axis=0)
        return exp/denominator
    
    def backprop(self,dl_dsum,learn_rate):
        # This method implements the deriavtive of the softbox and update rule for weight and bias
        
        for i ,gradiant in enumerate(dl_dsum):
            if gradiant == 0:
                continue
            
            t_exp=np.exp(self.last_sum)
            S=np.sum(t_exp)
            dout_dt = -t_exp[i] * t_exp / (S ** 2)
            dout_dt[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)  
            
            # Gradients of totals against weights/biases/input
            dt_dw = self.last_input
            dt_db = 1
            dt_dinputs = self.weights
            # Gradients of loss against totals#
            dL_dt = gradiant * dout_dt
            # Gradients of loss against weights/biases/input
            dL_dw = dt_dw[np.newaxis].T @ dL_dt[np.newaxis]
            dL_db = dL_dt * dt_db
            dL_dinputs = dt_dinputs @ dL_dt
            
            self.weights -= learn_rate * dL_dw
            self.bias -= learn_rate * dL_db
            return dL_dinputs.reshape(self.last_input_shape)
        
        
        