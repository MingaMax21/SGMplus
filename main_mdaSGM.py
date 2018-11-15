#!/usr/bin/env python3 -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:35:53 2018
Mono-Depth Adapted SGM
@author: Max Hoedel, 2018

"""
import numpy as np
import scipy as sp
from scipy import ndimage
#from PIL import Image as im
import matplotlib.pyplot as plt
import matplotlib.image as img
import csv

## Import images and calibration file
calib = []
with open('./data/Flowers-perfect/calib.txt', newline='') as inputfile:
    for row in csv.reader(inputfile):
        calib.append(row)
        
imL = img.imread('./data/Flowers-perfect/im0.png')
imR = img.imread('./data/Flowers-perfect/im1.png')

# Imread appears to normalize images, resolve to original 0-255 range (opt.):
imL[:,:,:] *= 255
imR[:,:,:] *= 255

# Convert to grayscale (retain int)
# r, g, b = imL[:,:,0], imL[:,:,1], imL[:,:,2]
imL = 0.2989 * imL[:,:,0] + 0.5870 * imL[:,:,1] + 0.1140 * imL[:,:,2]
imL = imL.astype(int)
imR = 0.2989 * imR[:,:,0] + 0.5870 * imR[:,:,1] + 0.1140 * imR[:,:,2]
imR = imR.astype(int)

# Downsample for processing (full, 1/2, 1/4)   Fnc: https://stackoverflow.com/questions/18666014/downsample-array-in-python
# !! Interpolation may not be correct downsampling method
imL = ndimage.interpolation.zoom(imL, 0.25)
imR = ndimage.interpolation.zoom(imR, 0.25)

# Display preprpcessed images:
plt.figure()
plt.imshow(imL,cmap='gray')
plt.show()
 
# Calculate raw cost ("use dynamic programming to force pixel matching "globally" along scanline)
# "make a decision about all matches along scanline at once"
# matching scanlines is important! "Matching by a path"
# From left to right, minimize cost SUM_x1->n c(x,y,d) along each scanline
# Scanline-wise disparity maps are generally streaky, consider homogeneity-criteria

## Define variables for SGM
r,c = np.shape(imL)         #dim
dLs = 15                    #discrete disparity levels (= mask size!)
#hom = 0                     #homogeneity fix toggle

# Determine raw cost
cost = np.zeros([dLs,r,c])
for d in range(dLs):
    for i in range(r):
        for j in range(c):
            #if (j)
            cost[d,i,j] = 1
            # Edge handling (crop left+right)
                
            # Create search template
        
            # Save cost value


