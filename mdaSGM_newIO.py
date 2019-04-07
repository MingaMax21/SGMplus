#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:28:34 2019
mdaSGM controller:
@author: max

input: L+R images, calib file, mono-depth estimations
output: disparity map as .pfm
"""
import mdaSGMlib as mda
import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
from skimage import color
from skimage import io
from skimage import img_as_ubyte
import os
import scipy.io as spio
# Start timer
start = time.time()
print('mdaSGM initialized\n')

#Debug
#imSet = ['Recycle']

# Full
imSet = ['Adirondack','ArtL','Jadeplant','Motorcycle','MotorcycleE','Piano','PianoL','Pipes','Playroom','Playtable','Recycle','Shelves','Teddy','Vintage']

# DEFINE CONSTANTS HERE:
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Block size for sum aggregation (good values: 5-9)
bS = 7 
bSf = np.float(bS)

# Penalty terms
p1 = (0.5 * bSf * bSf)
p2 = (2 * bSf * bSf)

# Number of paths (SUPPORTS 1-8)
nP = 8
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# Run algorithm for all images
for p in imSet:  
    
    print(p)
    # Ground truth (comparison only)
    gtL = mda.readGT(os.path.join("./data/", p, "disp0GT.pfm"))
    
    # Ground truth histogram
    gtL[gtL >= 1E308] = 0
    gtLhist, gtLbins = np.histogram(gtL, bins=256)
    gtLbins = gtLbins[0:gtLbins.size-1]
    
    # Save and remove gt mismatches from histogram
    gtFails = gtLhist[0]
    gtLhist[0] = 0
    
    # Read input images
    imL = color.rgb2gray(io.imread(os.path.join("./data", p, "im0.png")))
    imR = color.rgb2gray(io.imread(os.path.join("./data", p, "im1.png")))
    imL = img_as_ubyte(imL)
    imR = img_as_ubyte(imR)
    imL = imL.astype(np.int16)       
    imR = imR.astype(np.int16)
    
    # Read mono-depth images
    mdL = spio.loadmat(os.path.join("./data/", p, "im0/predict_depth.mat"))
    mdL = mdL["data_obj"]
    mdR = spio.loadmat(os.path.join("./data/", p, "im1/predict_depth.mat"))
    mdR = mdR["data_obj"]    
    
    # Calibration metrics for depth-disparity conversion
    cal = open(os.path.join("./data", p, "calib.txt"))
    focus, doffs, baseline, width, height, ndisp, vmin, vmax, dyavg, dymax = mda.readCal(cal) #cal.readlines()

    # Get disparity range dR from mono-depth metrics 
    #dR, dD = mda.dispRangeOld(mdL, mdR, doffs, baseline, focus)  #   Old: Activate for pure pixel disparity borders (worse)
    dR, dD = mda.dispRangeHist(mdL, mdR, doffs, baseline, focus)  #   New: Histogram based disparity borders (better)
         
    # ------------------------------ #
    
    #DEBUG: CHEAT DISP RANGE From INFO 
    #dR = np.arange(vmin, vmax)
    
    # ------------------------------ #
    
    # Calculate raw cost
    cIm = mda.rawCost(imL, imR, bS, dR)
    
    # Path search and cost aggregation
    lIm = mda.costAgg(cIm, p1, p2, nP)
    
    # Sum across paths
    S = np.sum(lIm, axis=3)
    
    # # Final disparity map as disparity value at location of minimum cost across all paths:
    dMap = np.argmin(S, axis=2) + dD
    
    # Output disp map
    fig,axes = plt.subplots(1,1)
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_title("SGM Disparity Map")
    axes.imshow(dMap,cmap='gray')
    plt.show()

    #Convert final disparities to .PFM for eval 
    dMap2 = np.flipud(dMap)
    dMap2 = dMap2.astype(np.float32)
    filename = os.path.join("./data", p, "disp0mda.pfm")
    file = open(filename,"w")
    dMap3 = mda.save_pfm(file,dMap2, scale = 1)
    file.close()

    # Remove zero values for conversion to depth
    dMap = dMap[np.isfinite(dMap)]
    dMap = dMap.reshape(height, width)

    # Depth map from disparity map
    dpMap = np.zeros((height, width))
    dpMap = ((baseline*focus)/(dMap + doffs)) / 1000
    
    # Output depth map
    fig,axes = plt.subplots(1,1)
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_title("SGM Depth Map")
    axes.imshow(dpMap,cmap='gray')
    plt.show()
    
    print("Time elapsed: --- %s seconds ---" % (time.time() - start))

    imageio.imsave(os.path.join("./data", p, "dMap.png"), dMap.astype(np.uint8))
    imageio.imsave(os.path.join("./data", p, "dpMap.png"), dpMap.astype(np.uint8))







