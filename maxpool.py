# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:55:44 2021

@author: Tharun
"""
import numpy as np

class maxpool:
    def __init__(self,poolsize):
        self.poolsize=poolsize
    
    def generate_regions(self,image):
        # Generator divides the image regions without over laping into 
        # poolsize window rigions
        # Input: image
        # Out : region
        
        h ,w ,c = image.shape
        new_h= int(h/self.poolsize)
        new_w=int(h/self.poolsize)
        
        for i in range(new_h):
            for j in range(new_w):
                im_region=image[(i*self.poolsize):(i*self.poolsize+self.poolsize),
                                (j*self.poolsize):(j*self.poolsize+self.poolsize)]
                yield im_region,i,j
        
    def forward(self,image):
        # Method calulates the max value in a window region of a image
        # Input = Image
        # Output = Downsampled or pooled image
        h,w,c=image.shape
        new_h=int(h/self.poolsize)
        new_w=int(w/self.poolsize)
        self.last_input=image
        output=np.zeros((new_h,new_w,c))
        for region,i,j in self.generate_regions(image):
            output[i,j]=np.amax(region,axis=(0,1))
            # amax is used here as we want the max or the array in (2d) the h and w 
            # not the channel axis
        return output
    
    def backprop(self,dl_dout):
        # Prviously we halved the image dimentionality now using back prop, we are
        # replacing the max value in the orignal image with the the graiant
        dl_din=np.zeros(self.last_input.shape)
        
        for region,i,j in self.generate_regions(self.last_input):
                  h, w, f = region.shape
                  amax=np.amax(region,axis=(0,1))
                  for i2 in range(h):
                      for j2 in range(w):
                          for f2 in range(f):
                              if region[i2,j2,f2]==amax[f2]:
                                  dl_din[i*self.poolsize+i2,j*self.poolsize+j2,f2]=dl_dout[i,j,f2]
        return dl_din
        
        
    
        

        