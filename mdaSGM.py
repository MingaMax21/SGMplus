#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:04:23 2018

"monocular depth assisted Semi-Global Matching (mdaSGM)"

@author: max hoedel, Technische Universitaet Muenchen, 2018
[credit: the base code of this program (no "mda") is based on the MATLAB code of Hirschmueller, 2008 (IEEE 30(2):328-341) ]
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.misc import imsave
from skimage import color
from skimage import io
from skimage import img_as_ubyte
from skimage import img_as_float32
from skimage import feature
from scipy import signal
from skimage import transform
import time

# Start timer
start = time.time()

# Create library for filenames (avoid having to change code to switch images)
#imSet = np.int(input('Please enter index of image set to be processed:\n[1]:Umbrella\n[2]:Piano\n[3]:Bust\n[4]:Recycle\n[5]:Motorcycle\n[6]:Mask\n\n:'))
imSet = 4

if   imSet == 1:
    imL = color.rgb2gray(io.imread("./data/Umbrella-perfect/im0.png"))
    imR = color.rgb2gray(io.imread("./data/Umbrella-perfect/im1.png"))
    dpL = io.imread("./data/Umbrella-perfect/im0_depth.png")
    dpR = io.imread("./data/Umbrella-perfect/im1_depth.png")
elif imSet == 2:
    imL = color.rgb2gray(io.imread("./data/Piano-perfect/im0.png"))
    imR = color.rgb2gray(io.imread("./data/Piano-perfect/im1.png"))
    dpL = io.imread("./data/Piano-perfect/im0_depth.png")
    dpR = io.imread("./data/Piano-perfect/im1_depth.png")
elif imSet == 3:
    imL = color.rgb2gray(io.imread("./data/mtlb_ex1/imL.png"))
    imR = color.rgb2gray(io.imread("./data/mtlb_ex1/imR.png"))
    dpL = io.imread("./data/mtlb_ex1/imL.png")
    dpR = io.imread("./data/mtlb_ex1/imL.png")
elif imSet == 4:
    imL = color.rgb2gray(io.imread("./data/Recycle-perfect/im0.png"))
    imR = color.rgb2gray(io.imread("./data/Recycle-perfect/im1.png"))
    dpL = io.imread("./data/Recycle-perfect/im0_depth.png")
    dpR = io.imread("./data/Recycle-perfect/im1_depth.png")
elif imSet == 5:
    imL = color.rgb2gray(io.imread("./data/Motorcycle-perfect/im0.png"))
    imR = color.rgb2gray(io.imread("./data/Motorcycle-perfect/im1.png"))
    dpL = io.imread("./data/Motorcycle-perfect/im0_depth.png")
    dpR = io.imread("./data/Motorcycle-perfect/im1_depth.png")
elif imSet == 6:
    imL = color.rgb2gray(io.imread("./data/Mask-perfect/im0.png"))
    imR = color.rgb2gray(io.imread("./data/Mask-perfect/im1.png"))
    dpL = io.imread("./data/Mask-perfect/im0_depth.png")
    dpR = io.imread("./data/Mask-perfect/im1_depth.png")
else:    
    sys.exit('Invalid entry! Program terminated!')

# Read Image and downsample to width = 360 and height = proportional
h,w = imL.shape
w2 = 360                        # ADJUST THIS PARAM FOR SCALING
h2 = np.int(w2*(h/w))
s2 = (h2,w2)
imL = transform.resize(imL,s2)
imR = transform.resize(imR,s2)
imL = img_as_ubyte(imL)
imR = img_as_ubyte(imR)
imL = imL.astype(np.int16)      # !!!TODO Dumb workaround to make integer subtraction on images possible 
imR = imR.astype(np.int16)

# Block size for sum aggregation
bS = 5 
bSf = np.float(5)

# Disparity range [-input,...,+input]
dR = 16                                 # !!!TODO Unintentionally hard coded, expand to 24 if possible
dR = np.arange(-dR,dR+1)
dR = dR.astype(np.int16)
R  = dR.shape[0]

# Penalty terms
p1 = (0.5 * bSf* bSf)
p2 = (2 * bSf* bSf)

# Number of paths (SUPPORTS 1-8)
nP = 4

# Plot imput images
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

# Plot raw mono-depth-images:
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("dpL")
axes.imshow(dpL,cmap='gray')
plt.show()

fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("dpR")
axes.imshow(dpR,cmap='gray')
plt.show()

# Resample mono-depth images: CAUTION: Scaling is an issue
hd,wd = dpL.shape
dpL = transform.resize(dpL,s2)  # (s2 defined above for input images)
dpR = transform.resize(dpR,s2)
dpL = img_as_ubyte(dpL)
dpR = img_as_ubyte(dpR)
dpL = dpL.astype(np.int16)      # !!!TODO Dumb workaround to make integer subtraction on images possible 
dpR = dpR.astype(np.int16)

# Plot resampled mono-depth images
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("dpL")
axes.imshow(dpL,cmap='gray')
plt.show()

fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("dpR")
axes.imshow(dpR,cmap='gray')
plt.show()

# !!!TODO: adequately preprocess images and adress scaling difference

# Calculate derivatives and look for jumps
dpL = dpL/255
dpR = dpR/255
edL = feature.canny(dpL, sigma = 1)
edR = feature.canny(dpR, sigma = 1)

# Plot edge images
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("edL")
axes.imshow(edL,cmap='gray')
plt.show()

fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("edR")
axes.imshow(edR,cmap='gray')
plt.show()

# TODO work on line continuity

# TODO create final edge map / list of "do-not-pass" points

def rawCost(imL, imR, bS, dR, R):    
    # initialize cost image (X x Y x disp_range)
    cIm =  np.zeros((imL.shape[0],imL.shape[1], R))
    cIm = cIm.astype(np.float)
    # initialize difference image (X x Y x 1)
    dIm = np.zeros((imL.shape[0], imL.shape[1]))
    dIm = dIm.astype(np.int16)     

    # Border constraints
    for i in dR:   
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
        dIm = np.abs(dIm)
        
        # Pre-convolution border handling (VERY VERY BAD, TODO!)
        # Left
        if i <= 0:
            lf1 = dIm[:,bL[0]]
            lf2 = ([lf1]*bL[0])
            lf = lf2*lf1
            dIm[:,0:bL[0]]  = np.sqrt(lf.T)
        
        else :
        # Right
            rt1 = dIm[:,bL[1]-1]
            rt2 = dIm.shape[1]-bL[1]
            rt3 = ([rt1]*rt2)
            rt = rt3 * rt1            
            dIm[:, bL[1]:bR[1]] = np.sqrt(rt.T)
        
        # calculate normalized sums (averages) with ones convolution
        flt = np.ones([bS,bS])
        cIm[:,:,i-dR[0]] = signal.convolve2d(dIm, flt, mode='same')/(bS*bS)
                
    return cIm
    
cIm = rawCost(imL, imR, bS, dR, R)

def diReMap(d, pind, dimX, dimY, dimD):
# parametrize line a1*y = a2*x +b, different parameters a1, a2, b for each direction

# Optimized boolean: Initialize all params,, rewrite as little as possible (still bad)
# Better idea? Use library to avoid excessive logicals?
    a1 = 1
    a2 = 0
    fo = 0
    
    if d == 0:
        a1 =1
        #a2 = 0 
        #fo = 0
        # x_inds = np.arange(0,dimX)                
        # y_inds = (a2*x_inds+pind)*a1        
        # inds_in = np.where(np.logical_and(y_inds>=0, y_inds < dimY))      
        
    elif d == 1:
        #a1 = 1
        a2 = 1
        #fo = 0
        # x_inds = np.arange(0,dimX)                
        # y_inds = (a2*x_inds+pind)*a1        
        # inds_in = np.where(np.logical_and(y_inds>=0, y_inds < dimY))
        
    elif d == 2:
        a1 = 0
        #a2 = 1
        #fo = 0
        # y_inds = np.arange(0,dimY)                 
        # x_inds = -1*pind*a2*np.ones(y_inds.shape[0])        
        # inds_in = np.where(np.logical_and(x_inds>=0, x_inds < dimX))
        
    elif d == 3:
        a1 = 1
        a2 = -1
        fo = 1
        # x_inds = np.arange(0,dimX)                
        # y_inds = (a2*x_inds+pind)*a1        
        # inds_in = np.where(np.logical_and(y_inds>=0, y_inds < dimY))     
        
    elif d == 4:
        #a1 = 1
        a2 = 0
        #fo = 1
        # x_inds = np.arange(0,dimX)                
        # y_inds = (a2*x_inds+pind)*a1        
        # inds_in = np.where(np.logical_and(y_inds>=0, y_inds < dimY))
        
    elif d == 5:
        #a1 = 1
        a2 = 1
        #fo = 1
        # x_inds = np.arange(0,dimX)                
        # y_inds = (a2*x_inds+pind)*a1        
        # inds_in = np.where(np.logical_and(y_inds>=0, y_inds < dimY))
        
    elif d == 6:
        a1 = 0
        #a2 = 1
        #fo = 1
        # y_inds = np.arange(0,dimY)                 
        # x_inds = -1*pind*a2*np.ones(y_inds.shape[0])        
        # inds_in = np.where(np.logical_and(x_inds>=0, x_inds < dimX))
        
    else:
        a1 = 1
        a2 = -1
        fo = 0
        # x_inds = np.arange(0,dimX)                
        # y_inds = (a2*x_inds+pind)*a1        
        # inds_in = np.where(np.logical_and(y_inds>=0, y_inds < dimY))      

    if a1 != 0:
        x_inds = np.arange(0,dimX)                
        y_inds = (a2*x_inds+pind)*a1        
        inds_in = np.where(np.logical_and(y_inds>=0, y_inds < dimY))        
                
    else:
        y_inds = np.arange(0,dimY)                 
        x_inds = -1*pind*a2*np.ones(y_inds.shape[0])        
        inds_in = np.where(np.logical_and(x_inds>=0, x_inds < dimX))
        
    x_inds = x_inds[inds_in[0]]
    y_inds = y_inds[inds_in[0]]
    
    x_inds = np.kron(np.ones((dimD,1)), x_inds).ravel()
    y_inds = np.kron(np.ones((dimD,1)), y_inds).ravel()
     
    if x_inds.shape[0] == 0:        
        z_inds = np.empty(0)
        
    else:
        help1 = np.arange(0,dimD) 
        help2 = np.int((x_inds.shape[0]/33))
        z_inds = np.kron(np.ones((help2,1)), help1).ravel('F')     

    # Flip based on direction
    if fo == 1:
        x_inds = np.flip(x_inds)
        y_inds = np.flip(y_inds)
    
    x_inds = x_inds.astype(np.int)
    y_inds = y_inds.astype(np.int)   
    z_inds = z_inds.astype(np.int)
       
    # Merge indices to inds matrix (tuple)   
    indMat = (y_inds,x_inds,z_inds)
    
    return indMat

def pathCost(slC, p1, p2):

    nL, nC = slC.shape   # labels, columns
    
    # constant grades matrix
    xx = np.arange(0,nL)
    XX,YY = np.meshgrid(xx,xx,sparse=False,indexing='ij')
    cGrad = np.zeros((nL,nL))
    
    # write values into cGrad matrix 
    cGrad[np.abs(XX-YY) == 1] = p1
    cGrad[np.abs(XX-YY) >  1] = p2
    
    # L slice matrix
    lrS = np.zeros(slC.shape)
    lrS[:,0] = slC[:,0]

    for c in range(1,nC):

        # calculate values C and M
        C  = slC[:,c]                          
        M1 = np.kron(np.ones((nL,1)), lrS[:,c-1])
        M  = np.amin(M1 + cGrad, axis=1);
        lrS[:,c] = C+M - np.amin(lrS[:,c-1])
        
    return(lrS)

def costAgg(cIm, p1, p2, nP):
# input: path slice of C (subC) , penalty terms
    dimY, dimX, dimD = cIm.shape
    dims = (dimY,dimX,dimD)
    dimMax = dimX + dimY
    dMax = np.arange(0,2*dimMax)    
    lIm = np.zeros((dimY, dimX, dimD, nP))
    
    # iterate over directions
    for d in range(nP):
        lIi = np.zeros(cIm.shape)
        print("--- %s seconds ---" % (time.time() - start))
        print("---step %s ----" % (d))
        # iterate over paths in direction
        for p in np.nditer(dMax):
            pind = p-dimMax-1            
            indMat = diReMap(d, pind, dimX, dimY, dimD)      
            inds = np.ravel_multi_index(indMat,dims,order='F') # Column-major indexing (Fortran style)
            slC = np.reshape(cIm[indMat], [int(inds.shape[0]/dimD)  , dimD],order='F')
            
            # If path exists:            
            if np.all(slC.shape) != 0:
                # evaluate cost
                lrS = pathCost(slC.T, p1, p2)
                # assign to output
                lIi[indMat]= lrS.flatten()
                lIm[:,:,:,d] = lIi
                         
    return lIm

lIm = costAgg(cIm, p1, p2, nP)

# Sum across paths
S = np.sum(lIm,axis=3)

# Final disparity map:
dMap = np.argmin(S,axis=2)+dR[0]

fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("Disparity Image")
axes.imshow(dMap,cmap='gray')
plt.show()

print("--- %s seconds ---" % (time.time() - start))

imsave('dMap.png',dMap)