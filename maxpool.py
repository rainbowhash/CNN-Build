# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:55:44 2021

@author: k20104661
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
        new_h= int(h//self.poolsize)
        new_w=int(h//self.poolsize)
        
        for i in range(new_h):
            for j in range(new_w):
                im_region=image[i*self.poolsize:i*self.poolsize+self.poolsize,
                                j*self.poolsize:j*self.poolsize+self.poolsize]
                yield im_region,i,j
        
    def forward(self,image):
        # Method calulates the max value in a window region of a image
        # Input = Image
        # Output = Downsampled or pooled image
        h,w,c=image.shape
        new_h=int(h//self.poolsize)
        new_w=int(w//self.poolsize)
        
        output=np.zeros((new_h,new_w,c))
        for region,i,j in self.generate_regions(image):
            output[i,j]=np.amax(region,axis=(0,1))
            # amax is used here as we want the max or the array in (2d) the h and w 
            # not the channel axis
        return output
            
        