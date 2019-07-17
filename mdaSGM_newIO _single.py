#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:28:34 2019
mdaSGM controller:
@author: max

input: L+R images, calib file, mono-depth estimations
output: disparity map as .pfm
"""
import mdaSGMlib_single as mda
import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
import gc
from skimage import color
from skimage import io
from skimage import img_as_ubyte
import os
import scipy.io as spio
# Start timer
start = time.time()
print('mdaSGM initialized\n')

# DEFINE IMAGE OR IMAGE SET HERE
# Debug
imSet = ['Teddy']
# Full
#imSet = ['Adirondack','ArtL','Jadeplant','Motorcycle','MotorcycleE','Piano','PianoL','Pipes','Playroom','Playtable','Recycle','Shelves','Teddy','Vintage']

# SELECT LAINA or LIU PREDICTIONS
pred = 'Laina'
#pred = 'Liu'

# DEFINE CONSTANTS HERE:
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Block size for sum aggregation (good values: 5-9)
bS = 5 
bSf = np.float(bS)

# Penalty terms
p1 = (0.5 * bSf * bSf)
p2 = (2 * bSf * bSf)

# Number of paths (SUPPORTS 1-8)
nP = 2
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# Run algorithm for all images
for p in imSet: 
    # Initialize log file
    log = p + '_log'    
    f = open('%s.txt' % log, 'w+')
    
    print(p)

    # Ground truth (comparison only)
    gtL = mda.readGT(os.path.join("./data/", p, "disp0GT.pfm"))
        
    # Read input images
    imL = color.rgb2gray(io.imread(os.path.join("./data", p, "im0.png")))
    imR = color.rgb2gray(io.imread(os.path.join("./data", p, "im1.png")))
    imL = img_as_ubyte(imL)
    imR = img_as_ubyte(imR)
    imL = imL.astype(np.int16)       
    imR = imR.astype(np.int16)
    
    # Read mono-depth images
    mdL = spio.loadmat(os.path.join("./data/", p, pred, "im0/predict_depth.mat"))
    mdR = spio.loadmat(os.path.join("./data/", p, pred, "im1/predict_depth.mat"))
    
    # Extract predicitions from mat file
    if pred == 'Liu':
        mdL = mdL["data_obj"]
        mdR = mdR["data_obj"]    
    elif pred == 'Laina':
        mdL = mdL["pred_depths"]
        mdR = mdR["pred_depths"]
    else:
        print('Invalid mono-depth method. Please select Liu or Laina')
        quit
            
    # Calibration metrics for depth-disparity conversion
    cal = open(os.path.join("./data", p, "calib.txt"))
    focus, doffs, baseline, width, height, ndisp, vmin, vmax, dyavg, dymax = mda.readCal(cal) #cal.readlines()

    # Get disparity range dR from mono-depth metrics 
    #dR, dD = mda.dispRangeOld(mdL, mdR, doffs, baseline, focus)  #   Old: Activate for pure pixel disparity borders (worse)
    dR, dD = mda.dispRangeHist(mdL, mdR, doffs, baseline, focus)  #   New: Histogram based disparity borders (better)
    print(dR)     
    # ------------------------------ #
    
    #DEBUG: CHEAT DISP RANGE From INFO 
    dR2 = np.arange(vmin, vmax)
    print(dR2)
    # ------------------------------ #
    
    # Calculate raw cost            ***   <- use [dR] for mda / [dr2] for cheat range    
    cIm = mda.rawCost(imL, imR, bS, dR2)
    
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

    # Depth map from disparity map (expressed in meters)
    dpMap = np.zeros((height, width))
    dpMap = ((baseline*focus)/(dMap + doffs))  / 1000
    
    # Output depth map
    fig,axes = plt.subplots(1,1)
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_title("SGM Depth Map")
    axes.imshow(dpMap,cmap='gray')
    plt.show()
    
    print("Time elapsed: --- %s seconds ---" % (time.time() - start))

    # Save output maps
    imageio.imsave(os.path.join("./data", p, "dMap.png"), dMap.astype(np.uint8))
    imageio.imsave(os.path.join("./data", p, "dpMap.png"), dpMap.astype(np.uint8))

    # Create Histograms:
    # GT Histogram
    gtL[gtL >= 1E308] = 0
    gtLhist, gtLbins = np.histogram(gtL, bins=vmax)
    
    # Save and remove gt mismatches from histogram
    gtFails = gtLhist[0]
    #gtLhist[0] = 0
    
    # mdaSGM output Histogram
    #gtL[gtL >= 1E308] = 0
    mdaHist, mdaBins = np.histogram(dMap, bins=dR[-1])
    

    gc.collect()




