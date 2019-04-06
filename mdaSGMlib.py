#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:57:15 2019

@author: max
"""


"""
Created on Wed Dec  5 14:04:23 2018

"monocular depth assisted Semi-Global Matching (mdaSGM)"

@author: max hoedel, Technische Universitaet Muenchen, 2018
[credit: the base code of this program (no "mda") is based on the MATLAB code of Hirschmueller, 2008 (IEEE 30(2):328-341) ]
"""
import numpy as np
import sys
import re
import time
import multiprocessing
from struct import unpack
from scipy import signal
# import matplotlib.pyplot as plt

# Start timer
start = time.time()

# Define function for reading GT disparities
def readGT(f):    
    with open(f,"rb") as f:
        type=f.readline().decode('latin-1')
        if "PF" in type:
            channels=3
        elif "Pf" in type:
            channels=1
        else:
            print("ERROR: Not a valid PFM file", file=sys.stderr)
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

    return(gt)

# Old pixel-based disparity range estimation from mono images
def dispRangeOld(mdL, mdR, doffs, baseline, focus):
        
    #mdL = signal.medfilt(mdL, kernel_size=9) # Old
    #mdR = signal.medfilt(mdR, kernel_size=9)
    
    mdL = mdL[np.nonzero(mdL)]
    mdR = mdR[np.nonzero(mdR)]
    
    # Min and max from monodepth images
    dminL = mdL.min()
    dmaxL = mdL.max()
    dminR = mdR.min()
    dmaxR = mdR.max()

    # Calculate mean min distance in MM for a-priori max disparity
    meanminZ = 1000*(dminL+dminR)/2

    # Calculate mean max distance in MM for a-priori min disparity
    meanmaxZ = 1000*(dmaxL+dmaxR)/2

    # Minimum disparity from minimum distance
    dMin = np.int(np.round((baseline*focus)/meanmaxZ - doffs))

    # Maximum disparity from minimum distance: disparity = (baseline*focal length)
    dMax = np.int(np.round((baseline*focus)/meanminZ - doffs))

    # Rounded and conservatively underestimated dMin

    # Disparity range NOTE: For calculation 
    dR = np.arange(dMin, dMax)        # disparity 0 is ignored, non plausible value (inf distance))
    dR = dR.astype(np.int16)
    
    # If dMin starting above 1: offset must be added back to dispImg 
    dD = dR[0] - 1
    
    return(dR, dD)
    
# Old pixel-based disparity range estimation from mono images
def dispRangeHist(mdL, mdR, doffs, baseline, focus):
        
   # mdL Histogram
    mdLhist, mdLbins = np.histogram(mdL, bins=256)    
    # convert mdL Histogram to pseudo-disp histogram and align
    mdLbins = (baseline*focus)/(mdLbins*1000) - doffs
        
    #flip histogram
    mdLhist = np.flip(mdLhist)
    mdLbins = np.flip(mdLbins)
    mdLbins = mdLbins[0:mdLbins.size-1]
    # Show histogram   
          
    # lower and upper 1% / 5% quantiles determine location of dispRange boundaries
    s1 = mdLhist.sum()    
    s=0
    ind=0
    brk = 0
    for n in mdLhist:
        s = s + n
        ind = ind+1
        if s >= 0.01 * s1 and brk == 0:
            dMin = np.int(np.round(mdLbins[ind]))
            brk = 1
        if s >= 0.99 * s1:
            dMax = np.int(np.round(mdLbins[ind]))
            break
        
    if dMin <1:
        dMin = 1
   
    # Disparity range NOTE: For calculation 
    dR = np.arange(dMin, dMax)        # disparity 0 is ignored, non plausible value (inf distance))
    dR = dR.astype(np.int16)
    
    # If dMin starting above 1: offset must be added back to dispImg 
    dD = dR[0] - 1
    
    # # Plots
    # fig,axes = plt.subplots(1,1)
    # plt.title("%s: Mono-Depth Estimation" % (p))
    # plt.bar(mdLbins, mdLhist)
    # plt.xlabel("Disparity [pix]")
    # plt.ylabel("Number of matches")
    # plt.grid(True)
    # plt.show()
    
    # fig,axes = plt.subplots(1,1)
    # plt.title("%s: Ground Truth" % (p))
    # plt.bar(gtLbins, gtLhist)
    # plt.xlabel("Disparity [pix]")
    # plt.ylabel("Number of matches")
    # plt.grid(True)
    # plt.show()
    
    return(dR, dD)    
    
# Read calibration file
def readCal(cal):
    # Read lines of file
    cam0, cam1, doffs, baseline, width, height, ndisp, isint, vmin, vmax, dyavg, dymax = cal.readlines()
     
    # Extract important metrics from calib file and convert to int/ float:
    doffs    = np.float32(doffs[6:])         # x-difference of principle points (doffs = cx1-cx0)
    baseline = np.float32(baseline[9:])      # camera baseline in mm
    focus    = cam0[6:]                      # focal length in pixels
    focus    = focus.split()
    focus    = np.float32(focus[0])
    width    = np.int(width[6:])             # img width in pix
    height   = np.int(height[7:])            # img height in pix
    ndisp    = np.int(ndisp[6:])             # conservative bound on number of disparity levels
    vmin     = np.int(vmin[5:])              # tight bound on min disparity
    vmax     = np.int(vmax[5:])              # tight bound on max disparity
    #   --> Floating point disparity to depth: Z = baseline * focal length / (dispVal + doffs)     
    return(focus, doffs, baseline, width, height, ndisp, vmin, vmax, dyavg, dymax)
    
# Raw cost aggregation    
def rawCost(imL, imR, bS, dR):     
    # initialize cost image (X x Y x disp_range)
    cIm =  np.zeros((imL.shape[0],imL.shape[1], dR.shape[0]))
    cIm = cIm.astype(np.float)
    # initialize difference image (X x Y x 1)
    dIm = np.zeros((imL.shape[0], imL.shape[1]))
    dIm = dIm.astype(np.int16)     

    for i in dR:   
          
        # Calculate borders        
        bL = np.array([1, imL.shape[1]]) - np.array([0, i])
        bR = bL + np.array([i,i])
           
        # Difference image within borders                
        dIm[:, bL[0] : bL[1]] = imR[:, bL[0]:bL[1]]  - imL[:, bR[0]:bR[1]]
        dIm = np.abs(dIm)
        
        # Pre-convolution border handling
        rt1 = dIm[:,bL[1]-1]
        rt2 = dIm.shape[1]-bL[1]
        rt3 = ([rt1]*rt2)
        rt  = rt3 * rt1            
        dIm[:, bL[1]:bR[1]] = np.sqrt(rt.T)
        
        # calculate normalized sums (averages) with ones convolution
        flt = np.ones([bS,bS])
        cIm[:,:,i-dR[0]] = signal.convolve2d(dIm, flt, mode='same')/(bS*bS)
                
    return cIm

# Directional remapping    
def diReMap(pind, dimX, dimY, dimD, par):
# NEW: Parameters are created outside this function to reduce excessive logicals
    # Valid for all paths except 3 and 7
    a1,a2,fo = par
    
    if a1 != 0:
        x_inds  = np.arange(0,dimX)                
        y_inds  = (a2*x_inds+pind)*a1        
        inds_in = np.where(np.logical_and(y_inds>=0, y_inds < dimY))        
    
    # Only paths 3 and 7. (expensive)        
    else:
        y_inds  = np.arange(0,dimY)                 
        x_inds  = pind*np.ones(y_inds.shape[0]) # (Oiginally -1* at front of term, as well as a2=1 term,) 
        inds_in = np.where(np.logical_and(x_inds>=0, x_inds < dimX))
            
    x_inds = np.kron(np.ones((dimD,1)), x_inds[inds_in[0]]).ravel()
    y_inds = np.kron(np.ones((dimD,1)), y_inds[inds_in[0]]).ravel()
     
    if x_inds.shape[0] == 0:        
        z_inds = np.empty(0)
        
    else:
        help1  = np.arange(0,dimD) 
        help2  = np.int((x_inds.shape[0]/dimD))
        z_inds = np.kron(np.ones((help2,1)), help1).ravel('F')     

    # Flip based on direction
    if fo == 1:
        x_inds = np.flip(x_inds)
        y_inds = np.flip(y_inds)
    
    # Merge indices to inds matrix (tuple)   
    indMat = (y_inds.astype(np.int) ,x_inds.astype(np.int), z_inds.astype(np.int))
    
    return indMat

# Evaluate path cost
def pathCost(slC, p1, p2):
    # labels, columns
    nL, nC = slC.shape   
    
    # constant grades matrix
    xx = np.arange(0,nL)
    XX,YY = np.meshgrid(xx,xx,sparse=True,indexing='ij') # sparsemess has no impact
    cGrad = np.zeros((nL,nL))
    
    # write values into cGrad matrix (penalty terms, homogeneity inforcement)
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

# Multiprocessing
def costProc(d, pars, cIm, return_dict, p1, p2):
    
    dimY, dimX, dimD = cIm.shape
    dims = (dimY,dimX,dimD)
    dimMax = dimX + dimY
    par = pars[d][:]
    lIi = np.zeros(cIm.shape)
    dimY,dimX,dimD = dims
    print("Path search process started: -%s-" % (d))
# case differentiation by path, each process is responsible for one direction
# Originally: static indR vector as np.arange(-dimX,dimMax) and logical test of slice length !=0 before evaluation
        
    if d == 0:
        indR = np.arange(0,dimMax-1)
    elif d == 1:
        indR = np.arange(0,dimY)
    elif d == 2:
        indR = np.arange(-dimX+1,dimY)
    elif d == 3:
        indR = np.arange(0,dimX)
    elif d == 4:
        indR = np.arange(0,dimMax-1)
    elif d == 5:
        indR = np.arange(0,dimY)
    elif d == 6:
        indR = np.arange(-dimX+1,dimY)
    else:
        indR = np.arange(0,dimX)
            
    for p in np.nditer(indR):
                          
        indMat = diReMap(p, dimX, dimY, dimD, par)      
        inds = np.ravel_multi_index(indMat,dims,order='F') # Column-major indexing (Fortran style)           
        slC = np.reshape(cIm[indMat], [int(inds.shape[0]/dimD) , dimD], order='F')            
        # If path exists: (now guaranteed through optimized indices)  old: if np.all(slC.shape) != 0: -> dostuff
        # evaluate cost
        lrS = pathCost(slC.T, p1, p2)
        # assign to output
        lIi[indMat] = lrS.flatten()
        
    return_dict[d] = lIi
 
# Cost agreggation              
def costAgg(cIm, p1, p2, nP):
# costAgg is a multi-process function, starting an instance of costProc for each path
    dimY, dimX, dimD = cIm.shape
    lIm  = np.zeros((dimY, dimX, dimD, nP))
    lIm2 = lIm
# input: path slice of C (subC) , penalty terms

    # Parameters for paths:
    pars = np.array([[1,-1,0], [1,0,0], [1,1,0], [0,1,0], [1,-1,1], [1,0,1], [1,1,1], [0,1,1]])
    #  8 Paths:       Up R      Hz R     Dn R     Vt Dn     Dn L      Hz L     Up L    Vt Up     
    # Parametrize line a1*y = a2*x +b, different parameters a1, a2, b for each direction
    # 'fo' flip-order term for left-looking paths, as well as up path
    
    # DEBUG: Additional path sets
    # pars = np.array([[0,1,1], [1,-1,0], [1,1,0], [1,-1,1], [1,0,1], [1,1,1], [0,1,1], [0,1,0]])
        
    # Multiprocessing:
    if __name__ != '__main__':
        processes = []
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        for d in range(nP):        
            p = multiprocessing.Process(target = costProc, args = (d, pars, cIm, return_dict, p1, p2))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
            print("Process %s joined" % (p))
        lIm = return_dict.values()        
        
        # workaround to get correct dim
        for d2 in range(len(lIm)):            
            lIm2[:,:,:,d2] = lIm[d2]
   
    return lIm2

# Write PFM file
def save_pfm(file, image, scale = 1): # source: [https://gist.github.com/chpatrick/8935738]
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)
  image.tofile(file)

