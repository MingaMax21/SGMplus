#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:04:23 2018

mono depth assisted Semi-Global Matching

@author: max hoedel
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage import color
from skimage import io
from skimage import img_as_ubyte
from scipy import signal

imL = color.rgb2gray(io.imread("./data/mtlb_ex1/imL.png"))      # ADAPT PATH
imR = color.rgb2gray(io.imread("./data/mtlb_ex1/imR.png"))      # ADAPT PATH
imL = img_as_ubyte(imL)
imR = img_as_ubyte(imR)
imL = imL.astype(np.int16)    # Dumb workaround to make integer subtraction on images possible (TODO!)
imR = imR.astype(np.int16)

# Block size for sum aggregation
bS = 5 #float(input("enter block size (odd): "))
bSf = np.float(5)

# Disparity range [-input,...,+input]
dR = 1 #float(input("enter +- disparity range: "))
dR = np.arange(-dR,dR+1)
dR = dR.astype(np.int16)
R  = dR.shape[0]

# Penalty terms
p1 = (0.5 * bSf* bSf)
p2 = (2 * bSf* bSf)

# Number of paths
nP = 8

fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("imL")
axes.imshow(imL,cmap='gray')
plt.show()

fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("imR")
axes.imshow(imR,cmap='gray')
plt.show()

def rawCost(imL, imR, bS, dR, R):
    # input: 2 images, matching-block size, disp_range vector, disp_range val
    # Calculates raw cost and convoluted block cost for 2 images across range
    
    # initialize cost image (X x Y x disp_range)
    cIm =  np.zeros((imL.shape[0],imL.shape[1], R))
    cIm = cIm.astype(np.int16)
    # initialize difference image (X x Y x 1)
    dIm = np.zeros((imL.shape[0], imL.shape[1]))
    dIm = dIm.astype(np.int16)     

    # Border constraints
    #for i in range(0):
    #i = -16
    #while i ==-16:
    for i in dR: # == 5:    
        if i < 0:
            l = i
            r = 0
        elif i > 0:
            l = 0
            r = i
        else:
            l = 0
            r = 0
        
        # Calculate borders
        bL = np.array([1, imL.shape[1]]) - np.array([l, r])  
        bR = bL + np.array([i,i])
           
        # Difference image within borders                
        dIm[:, bL[0] : bL[1]] = imR[:, bL[0]:bL[1]]  - imL[:, bR[0]:bR[1]]
        dIm = abs(dIm)
        
        # Do border handling (VERY VERY BAD, TODO!)
        # Left
        if i <= 0:
            dog = dIm[:,bL[0]]
            cat = ([dog]*bL[0])
            mag = cat*dog
            dIm[:,0:bL[0]]  = np.sqrt(mag.T)
        
        else :
        # Right
            dog2 = dIm[:,bL[1]-1]
            lol = dIm.shape[1]-bL[1]
            cat2 = ([dog2]*lol)
            mag2 = cat2 * dog2
            #dog2 = dIm[:,bL[1]-1] * np.ones(1,dIm.shape[1] - bL[1]+1) 
            dIm[:, bL[1]:bR[1]] = np.sqrt(mag2.T)
        
        # calculate sums with ones convolution
        flt = np.ones([bS,bS])
        #print(i+dR[0])
        cIm[:,:,i-dR[0]] = signal.convolve2d(dIm, flt, mode='same')
        
        # normalize

        #i+=1
    return dIm, cIm
    
dIm, cIm = rawCost(imL, imR, bS, dR, R)

def diReMap(d, pind, dimX, dimY, dimD):
# parametrize lline a1*y = a2*x +b
# different parameters a1, a2, b for each direction
# fo is parameter for negating signs, pointing in opposite direction    
    if d == 0:
        a1 =1
        a2 = 0 
        fo = 0
    elif d == 1:
        a1 = 1
        a2 = 1
        fo = 0
    elif d == 2:
        a1 = 0
        a2 = 1
        fo = 0
    elif d == 3:
        a1 = 1
        a2 = -1
        fo = 1
    elif d == 4:
        a1 = 1
        a2 = 0
        fo = 1
    elif d == 5:
        a1 = 1
        a2 = 1
        fo = 1
    elif d == 6:
        a1 = 0
        a2 = 1
        fo = 1
    else:
        a1 = 1
        a2 = -1
        fo = 0    

    if a1 != 0:
        x_inds = np.arange(0,dimX)                       
        y_inds = (a2*x_inds.T+pind)*a1
        inds_in = np.where(np.logical_and(y_inds>0, y_inds <= dimY))
        inds_in = inds_in[0]
                
    else:

        y_inds = np.arange(0,dimY)        
        x_inds = -1*pind*a2* np.ones(y_inds.shape[0])
        inds_in = np.where(np.logical_and(x_inds>0, x_inds <= dimX)) 
        inds_in = inds_in[0]
        
    x_inds = x_inds[inds_in]
    print(x_inds.shape)
    print(x_inds)
    y_inds = y_inds[inds_in]
   
    # !!! up to here no issues !!! 
    
    slLen = x_inds.shape[0]
    x_inds = np.kron(np.ones((slLen*dimD,1)), x_inds).ravel()                   #repmat(x_inds, 1, size_d)'
    print(x_inds.shape)
    print(x_inds)
    y_inds = np.kron(np.ones((x_inds.shape[0]*dimD,1)),y_inds)
    
    help1 = np.arange(0,dimD)
    # z_inds = np.kron(np.ones((slLen,1)), help1)
    # z_inds = z_inds.ravel(1)

    
    # Flip based on direction
    if fo == 1:
        x_inds = np.fliplr(x_inds)
        y_inds = np.fliplr(y_inds)
        
    # Merge indices to inds vector
    
    inds = y_inds
    
    return inds_in, inds

def pathCost(slice1, p1, p2):
    gSlice = 0
    return(gSlice)


def costAgg(cIm, p1, p2, nP):
# input: path slice of C (subC) , penalty terms
    dimY, dimX, dimD = cIm.shape
    dimMax = dimX + dimY
    
    lIm = np.zeros((dimX, dimY, dimD, nP))
    
    # iterate over directions
    for d in range(nP):
        lIi = np.zeros(cIm.shape)
        
        # iterate over paths in direction
        for p in range(2*dimMax):
            pind = p-dimMax
            inds, inds_in = diReMap(d, pind, dimX, dimY, dimD)
            #print(inds_in.shape)
            #slice = reshape(cIm(inds), [inds.shape[0] / dimD.shape[0], dimD.shape[0]])
            #print(pind)
            # extract path from cost array
            
    
    return lIi

lIi = costAgg(cIm, p1, p2, nP)












