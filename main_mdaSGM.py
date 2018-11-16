#!/usr/bin/env python3 -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:35:53 2018
Mono-Depth Adapted SGM
@author: Max Hoedel, 2018

"""
import sys
import numpy as np
#import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as img
import csv

## Import images and calibration file
calib = []
with open('calib.txt', newline='') as inputfile:
    for row in csv.reader(inputfile):
        calib.append(row)
        
imL = img.imread('im0.png')
imR = img.imread('im1.png')

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
imL = ndimage.interpolation.zoom(imL, 0.125)
imR = ndimage.interpolation.zoom(imR, 0.125)

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
r,c = np.shape(imL)         
dLs = 125                # 125 should cover most movement

# Enforce odd mask size
if dLs % 2 == 0:
    print('ERROR: mask Size ("dLs") must be odd!')
    sys.exit(1)
crop = int((dLs-1)/2) # Edge handling parameter

# Determine raw pixel-wise cost and initial disparity estimate
cost = np.zeros([r,c,dLs])
cost = cost.astype(int)
minCost = np.zeros([r,c])
minCost = minCost.astype(int)
dispEst = np.zeros([r,c])
dispEst = dispEst.astype(int)

for i in range(r):
    for j in range(crop , c-crop):
        for d in range(dLs):            
            cost[i,j,d] = abs(imL[i,j] - imR[i,j - crop + d])          
              
        # Gather minimums for disparity estimation  
        minCost[i,j] = min(cost[i,j])
        
        # Gather index of minumum,         
        ind1 = np.unravel_index(np.argmin(cost[i,j]), cost[i,j].shape)[0]
              
        # calc and append disparity estimate
        dispEst[i,j] = abs(crop -ind1)
   
## TODO Trim and normalize result maps

plt.figure()
plt.imshow(minCost,cmap='gray')
plt.show()     

plt.figure()
plt.imshow(dispEst,cmap='gray')
plt.show()   
 