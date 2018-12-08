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
dR = 16 #float(input("enter +- disparity range: "))
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
    # initialize cost image (X x Y x disp_range)
    cIm =  np.zeros((imL.shape[0],imL.shape[1], R))
    cIm = cIm.astype(np.float)
    # initialize difference image (X x Y x 1)
    dIm = np.zeros((imL.shape[0], imL.shape[1]))
    dIm = dIm.astype(np.int16)     

    # Border constraints
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
        flt = np.ones([bS,bS])/(bS*bS)
        cIm[:,:,i-dR[0]] = signal.convolve2d(dIm, flt, mode='same')
                
    return dIm, cIm
    
dIm, cIm = rawCost(imL, imR, bS, dR, R)

def diReMap(d, pind, dimX, dimY, dimD):
# parametrize line a1*y = a2*x +b, different parameters a1, a2, b for each direction
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
        inds_in = np.where(np.logical_and(y_inds>=0, y_inds < dimY))        
        inds_in = inds_in[0]      
                
    else:
        y_inds = np.arange(0,dimY)                 
        x_inds = -1*pind*a2*np.ones(y_inds.shape[0]) # off by 2        
        inds_in = np.where(np.logical_and(x_inds>=0, x_inds < dimX))
        inds_in = inds_in[0]        
        
    x_inds = x_inds[inds_in]
    y_inds = y_inds[inds_in]
    
    x_inds = np.kron(np.ones((dimD,1)), x_inds).ravel()
    y_inds = np.kron(np.ones((dimD,1)), y_inds).ravel()
     
    if x_inds.shape[0] == 0:        
        z_inds = np.empty(0)
        
    else:
        help1 = np.arange(0,dimD) 
        help2 = int((x_inds.shape[0]/33))
        z_inds = np.kron(np.ones((help2,1)), help1).ravel('F')     
    
#    Flip based on direction
    if fo == 1:
        x_inds = np.flip(x_inds)
        y_inds = np.flip(y_inds)
    
    x_inds = x_inds.astype(int)
    y_inds = y_inds.astype(int)   
    z_inds = z_inds.astype(int)
       
    # Merge indices to inds tuple    
    indMat = (y_inds,x_inds,z_inds)
    # Create dims tuple    
    dims = (dimY, dimX, dimD)
    inds = np.ravel_multi_index(indMat,dims,order='F') # Column-major indexing
    
    return inds, indMat

def pathCost(slC, p1, p2):

    nL = slC.shape[0]   # labels
    nC = slC.shape[1]   # columns
    
    # constant grades matrix
    xx = np.arange(0,nL)
    XX,YY = np.meshgrid(xx,xx,sparse=False,indexing='ij')
    cGrad = np.zeros((nL,nL))
    
    #write values into cGrad matrix 
    cGrad[abs(XX-YY) == 1] = p1
    cGrad[abs(XX-YY) > 1] = p2
    
    #L slice matrix
    lrS = np.zeros(slC.shape)
    lrS[:,0] = slC[:,0]

    for c in range(1,nC):
        # save previous slice
        lrSlast = lrS[:,c-1]
        # calculate values C and M
        C = slC[:,c]                          
        M1 = np.kron(np.ones((nL,1)), lrSlast.T)
        M = np.amin(M1 + cGrad, axis=1);
        lrS[:,c] = C+M - np.amin(lrSlast)
        
    return(lrS)

def costAgg(cIm, p1, p2, nP):
# input: path slice of C (subC) , penalty terms
    dimY, dimX, dimD = cIm.shape
    dimMax = dimX + dimY    
    lIm = np.zeros((dimY, dimX, dimD, nP))
    
    # iterate over directions
    for d in range(nP):
        lIi = np.zeros(cIm.shape)
        
        # iterate over paths in direction
        for p in range(2*dimMax):
            pind = p-dimMax-1
            inds, indMat = diReMap(d, pind, dimX, dimY, dimD)            
            h1 = cIm[indMat]
            h2 = int(inds.shape[0]/dimD)
            slC = np.reshape(h1, [h2, dimD],order='F')

            # If path exists:
            if np.all(slC.shape) != 0:
                # evaluate cost
                lrS = pathCost(slC.T, p1, p2)
                # assign to output
                lIi[indMat] = lrS.flatten()
                lIm[:,:,:,d] = lIi
                
    return lIm

lIm = costAgg(cIm, p1, p2, nP)

def dispMap(lIm, dR):
    
    S = np.sum(lIm,axis=3)
    dMap = np.argmin(S,axis=2)+dR[0]

    return dMap

dMap = dispMap(lIm, dR)

fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("Disparity Image")
axes.imshow(dMap,cmap='gray')
plt.show()
