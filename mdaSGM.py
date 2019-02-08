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
import re
import imageio
import scipy.io as spio
from struct import *
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
print('mdaSGM initialized\n')

# Create library for filenames (avoid having to change code to switch images)
imSet = np.int(input('Please enter index of image set to be processed:\n[1]:Motorcycle\n[2]:Piano\n[3]:Recycle\n\n:'))
#imSet = 3 #(debug)

# Define function for reading GT disparities
def readGT(f):    
    with open(f,"rb") as f:
        type=f.readline().decode('latin-1')
        if "PF" in type:
            channels=3
        elif "Pf" in type:
            channels=1
        else:
            print("ERROR: Not a valid PFM file",file=sys.stderr)
            sys.exit(1)
    # Line 2: width height
        line=f.readline().decode('latin-1')
        width,height=re.findall('\d+',line)
        width=int(width)
        height=int(height)
    # Line 3: +ve number means big endian, negative means little endian
        line=f.readline().decode('latin-1')
        BigEndian=True
        if "-" in line:
            BigEndian=False
    # Slurp all binary data
        samples = width*height*channels;
        buffer  = f.read(samples*4)
    # Unpack floats with appropriate endianness
        if BigEndian:
            fmt=">"
        else:
            fmt="<"
            fmt= fmt + str(samples) + "f"
            img = np.array(unpack(fmt,buffer))
            
    # Modified: reshape tuple back into 2D image
        gt = np.flipud(img.reshape(height,width))
        #gt = gt[np.isfinite(gt)]
    # replace inf with nan to allaw for max calculation
        #gt[~np.isfinite(gt)] = np.nan
        #gt = np.ma.masked_array(gt, ~np.isfinite(gt)).filled(np.nan)
          
    return(gt)
    
# Import L+R image pair, calib file, as well as metric depth maps
if   imSet == 1:
    imL = color.rgb2gray(io.imread("./data/Motorcycle-perfect/im0_resized.png"))
    imR = color.rgb2gray(io.imread("./data/Motorcycle-perfect/im1_resized.png"))
    dpL = spio.loadmat("./data/Motorcycle-perfect/im0_results.mat")
    dpL = dpL["pred_depths"]
    dpR = spio.loadmat("./data/Motorcycle-perfect/im1_results.mat")
    dpR = dpR["pred_depths"]
    gtL = readGT("./data/Motorcycle-perfect/disp1.pfm")
    gtR = readGT("./data/Motorcycle-perfect/disp1.pfm")
    cal = open('./data/Motorcycle-perfect/calib.txt', 'r')

elif imSet == 2:
    imL = color.rgb2gray(io.imread("./data/Piano-perfect/im0_resized.png"))
    imR = color.rgb2gray(io.imread("./data/Piano-perfect/im1_resized.png"))
    dpL = spio.loadmat("./data/Piano-perfect/im0_results.mat")
    dpL = dpL["pred_depths"]
    dpR = spio.loadmat("./data/Piano-perfect/im1_results.mat")
    dpR = dpR["pred_depths"]
    gtL = readGT("./data/Piano-perfect/disp0.pfm")
    gtR = readGT("./data/Piano-perfect/disp1.pfm")
    cal = open('./data/Piano-perfect/calib.txt', 'r')
    
elif imSet == 3:
    imL = color.rgb2gray(io.imread("./data/Recycle-perfect/im0_resized.png"))
    imR = color.rgb2gray(io.imread("./data/Recycle-perfect/im1_resized.png"))
    dpL = spio.loadmat("./data/Recycle-perfect/im0_results.mat")
    dpL = dpL["pred_depths"]
    dpR = spio.loadmat("./data/Recycle-perfect/im1_results.mat")
    dpR = dpR["pred_depths"]
    gtL = readGT("./data/Recycle-perfect/disp0.pfm")
    gtR = readGT("./data/Recycle-perfect/disp1.pfm")
    cal = open('./data/Recycle-perfect/calib.txt', 'r')
    
#elif imSet == 4:
    # imL = color.rgb2gray(io.imread("./data/Umbrella-perfect/im0_resized.png"))
    # imR = color.rgb2gray(io.imread("./data/Umbrella-perfect/im1_resized.png"))
    # dpL = io.imread("./data/Umbrella-perfect/im0_depth.png")
    # dpR = io.imread("./data/Umbrella-perfect/im1_depth.png")
    
#elif imSet == 5:
    # imL = color.rgb2gray(io.imread("./data/Mask-perfect/im0_resized.png"))
    # imR = color.rgb2gray(io.imread("./data/Mask-perfect/im1_resized.png"))
    # dpL = spio.loadmat("./data/Mask-perfect/im0_results.mat")
    # dpL = dpL["pred_depths"]
    # dpR = spio.loadmat("./data/Mask-perfect/im1_results.mat")
    # dpR = dpR["pred_depths"]
    
#elif imSet == 6: 
    # imL = color.rgb2gray(io.imread("./data/mtlb_ex1/imL.png"))
    # imR = color.rgb2gray(io.imread("./data/mtlb_ex1/imR.png"))
    # dpL = io.imread("./data/mtlb_ex1/imL.png")
    # dpR = io.imread("./data/mtlb_ex1/imL.png")
    
else:    
    sys.exit('Invalid entry! Program terminated!')

# Plot imput images
print("Input image left:\n")
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("imL")
axes.imshow(imL,cmap='gray')
plt.show()

print("Input image right:\n")
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("imR")
axes.imshow(imR,cmap='gray')
plt.show()

# Plot raw mono-depth-images:
print("Mono-depth image left:\n")
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("dpL")
axes.imshow(dpL,cmap='gray')
plt.show()

print("Mono-depth image right:\n")
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("dpR")
axes.imshow(dpR,cmap='gray')
plt.show()

# Plot ground truth disparities
print("Ground truth disp left:\n")
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("gtL")
axes.imshow(gtL,cmap='gray')
plt.show()

print("Ground truth disp left:\n")
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("gtR")
axes.imshow(gtR,cmap='gray')
plt.show()

print("Calculating ground truth depth maps...please wait")

# Read and split calibration file
cam0, cam1, doffs, baseline, width, height, ndisp, isint, vmin, vmax, dyavg, dymax = cal.readlines()

# Extract important metrics from calib file and convert to int/ fload:
doffs    = np.float32(doffs[6:])         # x-difference of principle points (doffs = cx1-cx0)
baseline = np.float32(baseline[9:])      # camera baseline in mm
focus    = np.float32(cam0[6:13])        # focal length in pixels
width    = np.int(width[6:])             # img width in pix
height   = np.int(height[7:])            # img height in pix
ndisp    = np.int(ndisp[6:])             # conservative bound on number of disparity levels
vmin     = np.int(vmin[5:])              # tight bound on min disparity
vmax     = np.int(vmax[5:])              # tight bound on max disparity
#   --> Floating point disparity to depth: Z = baseline * focal length / (dispVal + doffs)

# Scaling factors and resized dimensions
width2 = imL.shape[1]
height2 = imL.shape[0]
scale = width2/width
doffs2 = scale*doffs
focus2 = scale*focus

# Workaround to make integer subtraction on images possible
# Resized images from network are [304x228]    
imL = img_as_ubyte(imL)
imR = img_as_ubyte(imR)
imL = imL.astype(np.int16)       
imR = imR.astype(np.int16)

# Extract important metrics from depth matrices:
dminL = dpL.min()
dmaxL = dpL.max()
dminR = dpR.min()
dmaxR = dpR.max()
dvarL = dmaxL - dminL
dvarR = dmaxR - dminR

# Calculate mean min distance in MM for a-priori max disparity
meanminZ = 1000*(dminL+dminR)/2

# maximum disparity from minimum distance: disparity = (baseline*focal length)/depth - doffs
dRange = ((baseline*focus)/meanminZ - doffs)*scale

# Debugging: extract equivalent metrics from GT, calculate depth map, compare for plausibility
gtminL = gtL[np.isfinite(gtL)]
gtminL = gtminL.min()
gtminR = gtR[np.isfinite(gtR)]
gtminR = gtminR.min()
gtmaxL = gtL[np.isfinite(gtL)]
gtmaxL = gtmaxL.max()
gtmaxR = gtR[np.isfinite(gtR)]
gtmaxR = gtmaxR.max()
gtvarL = gtmaxL - gtminL
gtvarR = gtmaxR - gtminR

# Create depthmap from GT disparity image 

gtdMapL = np.zeros((height, width))
gtdMapL = ((baseline*focus)/(gtL+doffs)) / 1000
        
gtdMapR = np.zeros((height, width))
gtdMapR = ((baseline*focus)/(gtR+doffs)) / 1000     

gtdminL = gtdMapL.min()
gtdminR = gtdMapR.min()
gtdmaxL = gtdMapL.max()
gtdmaxR = gtdMapR.max()
gtdvarL = gtdmaxL - gtdminL
gtdvarR = gtdmaxR - gtdminR

#mediandp = np.median(gtdMapL)
#histdp = plt.hist([gtdMapL])

print("Ground truth depth left:\n")
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("gtdMapL")
axes.imshow(gtdMapL,cmap='gray')
#plt.colorbar('gray')
plt.show()

print("Ground truth depth right:\n")
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("gtdMapL")
axes.imshow(gtdMapR,cmap='gray')
plt.show()        

# Outputs before SGM calculation:
print("Size of original input image: %s x %s \n" % (width, height))
print("Sized of NN resized image: %s x %s \n" % (dpL.shape[1], dpL.shape[0]))
print("Mono-depth range estimation (left): %s [m] \n" % (dvarL))
print("Mono-depth range estimation (right): %s [m] \n" % (dvarR))
print("Mono-depth disparity range estimation: %s [pix] \n" % (dRange))
print("Ground truth disparity range (left): %s [pix] \n" % (gtvarL))
print("Ground truth disparity range (right): %s [pix] \n" % (gtvarL))  
print("Ground truth depth range (left): %s [m] \n" % (gtdvarL))
print("Ground truth depth range (right): %s [m] \n" % (gtdvarL)) 

# Block size for sum aggregation
bS = 5 
bSf = np.float(bS)

# Disparity range [-input,...,+input]
dR = 16                                 # !!!TODO Unintentionally hard coded, expand dynamically
dR = np.arange(-dR,dR+1)
dR = dR.astype(np.int16)
R  = dR.shape[0]

# Penalty terms
p1 = (0.5 * bSf* bSf)
p2 = (2 * bSf* bSf)

# Number of paths (SUPPORTS 1-8)
nP = 4

# # Resample mono-depth images: CAUTION: Scaling is an issue
# hd,wd = dpL.shape
# dpL = transform.resize(dpL,s2)  # (s2 defined above for input images)
# dpR = transform.resize(dpR,s2)
# dpL = img_as_ubyte(dpL)
# dpR = img_as_ubyte(dpR)
# dpL = dpL.astype(np.int16)
# dpR = dpR.astype(np.int16)

# !!!TODO: adequately preprocess images and optimize filter

# Calculate derivatives and look for jumps (ADAPTIVE SIGMA!)
edL = feature.canny(dpL, sigma = 7)
edR = feature.canny(dpR, sigma = 7)

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

# Optimized boolean: Initialize all params,, rewrite as little as possible
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
            print(p)
            pind = p-dimMax-1            
            indMat = diReMap(d, pind, dimX, dimY, dimD)      
            inds = np.ravel_multi_index(indMat,dims,order='F') # Column-major indexing (Fortran style) # !!! PROBLEM
            
           # inds2 = 
           # Step 0:
           # p0-532   inds -> 0
           # p533-760 inds -> 10032
           # p761-1063 inds -> 0
           # Step 1:
           # p0-229  inds -> 0
           # p230-457  inds -> increase from 33 to 7524 in increments of 33
           # p458-533 inds -> 7524
           # p534-760 infd  -> decrease from 7524 to 33 in increments of 33
           #p761-1063  inds ->0
           # Step 2:
           # p0-1063 inds -> 7524
           # Step 3: 
           # p0-532   inds -> 0
           
           
            print(inds.shape[0])
            slC = np.reshape(cIm[indMat], [int(inds.shape[0]/dimD) , dimD],order='F')
            #slC = np.reshape(cIm[indMat], [int(((dimD+1)*dimY-dimY)/dimD) , dimD],order='F')
            
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

# Remove zero values
dMap = dMap[np.isfinite(dMap)]
dMap = dMap.reshape(height2,width2)

# !!! TODO fix dRange. Temp debug fix: force to 0
dMap = np.abs(dMap)


# Depth map from disparity map
dpMap = np.zeros((height2, width2))
dpMap = ((baseline*focus2)/(dMap + doffs2)) / 1000

print("SGM Disparity map:\n")
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("SGM Disparity Image")
axes.imshow(dMap,cmap='gray')
plt.show()

print("SGM Depth map:\n")
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("SGM Depth Image")
axes.imshow(dpMap,cmap='gray')
plt.show()

print("--- %s seconds ---" % (time.time() - start))

imageio.imsave('dMap.png',dMap.astype(np.uint8))