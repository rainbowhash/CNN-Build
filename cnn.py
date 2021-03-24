# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:11:08 2021

@author: k20104661
"""

from conv import conv3x3
import mnist

train_image=mnist.train_images()
test_image=mnist.test_image()

conv=conv3x3(8)
out=conv.forward_pass(train_image[0])
print(out)