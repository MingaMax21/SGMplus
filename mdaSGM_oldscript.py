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
import time
import multiprocessing
from struct import unpack
from skimage import color
from skimage import io
from skimage import img_as_ubyte
from skimage import feature
from scipy import signal
from skimage import transform
import scipy.io as spio
from matplotlib import pyplot as plt 
from PIL import Image



# Start timer
start = time.time()
print('mdaSGM initialized\n')

# Create library for filenames (avoid having to change code to switch images)
imSet = np.int(input('Please enter index of image set to be processed:\n[1]:Motorcycle\n[2]:Piano\n[3]:Recycle\n\n:'))
#imSet = 3 #(debug)

# 
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
        
else:    
    sys.exit('Invalid entry! Program terminated!')

# # Plot imput images
# print("Input image left:\n")
# fig,axes = plt.subplots(1,1)
# axes.set_xlabel("X")
# axes.set_ylabel("Y")
# axes.set_title("imL")
# axes.imshow(imL,cmap='gray')
# plt.show()

# print("Input image right:\n")
# fig,axes = plt.subplots(1,1)
# axes.set_xlabel("X")
# axes.set_ylabel("Y")
# axes.set_title("imR")
# axes.imshow(imR,cmap='gray')
# plt.show()

# # Plot raw mono-depth-images:
# print("Mono-depth image left:\n")
# fig,axes = plt.subplots(1,1)
# axes.set_xlabel("X")
# axes.set_ylabel("Y")
# axes.set_title("dpL")
# axes.imshow(dpL,cmap='gray')
# plt.show()

# print("Mono-depth image right:\n")
# fig,axes = plt.subplots(1,1)
# axes.set_xlabel("X")
# axes.set_ylabel("Y")
# axes.set_title("dpR")
# axes.imshow(dpR,cmap='gray')
# plt.show()

# # Plot ground truth disparities
# print("Ground truth disp left:\n")
# fig,axes = plt.subplots(1,1)
# axes.set_xlabel("X")
# axes.set_ylabel("Y")
# axes.set_title("gtL")
# axes.imshow(gtL,cmap='gray')
# plt.show()

# print("Ground truth disp left:\n")
# fig,axes = plt.subplots(1,1)
# axes.set_xlabel("X")
# axes.set_ylabel("Y")
# axes.set_title("gtR")
# axes.imshow(gtR,cmap='gray')
# plt.show()

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

print(focus)
#Calculate mono-depth metrics:
# Scaling factors and resized dimensions
width2  = imL.shape[1]
width3  = dpL.shape[1]
height2 = imL.shape[0]
scale   = width2/width # Scale for resized imgs
scale2  = width3/width # Scale for monodepth map
doffs2  = scale*doffs
focus2  = scale*focus

# Workaround to make integer subtraction on images possible
# Resized images from network are [304x228]    
imL = img_as_ubyte(imL)
imR = img_as_ubyte(imR)
imL = imL.astype(np.int16)       
imR = imR.astype(np.int16)

# Disparity range estimation:
dminL = dpL.min()
dmaxL = dpL.max()
dminR = dpR.min()
dmaxR = dpR.max()
dvarL = dmaxL - dminL
dvarR = dmaxR - dminR

# Check plausibility

# Calculate mean min distance in MM for a-priori max disparity
meanminZ = 1000*(dminL+dminR)/2

# Calculate mean max distance in MM for a-priori min disparity
meanmaxZ = 1000*(dmaxL+dmaxR)/2

# Minimum disparity from maximum distance
dMin = np.int(np.round(((baseline*focus)/meanmaxZ - doffs2) * scale2))

# Maximum disparity from minimum distance: disparity = (baseline*focal length)/depth - doffs
dRange = np.int(np.round(((baseline*focus)/meanminZ - doffs2) * scale2))

# Calculate derivatives and look for jumps
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

# Scaled GT disparity metrics
gtsminL = gtminL*scale
gtsminR = gtminR*scale
gtsmaxL = gtmaxL*scale
gtsmaxR = gtmaxR*scale

gtsmin = np.round((gtsminL + gtsminR) / 2)
gtsmax = np.round((gtsmaxL + gtsmaxR) / 2)

# Create depthmap from GT disparity image
gtdMapL = np.zeros((height, width))
gtdMapL = ((baseline*focus)/(gtL+doffs)) / 1000
        
gtdMapR = np.zeros((height, width))
gtdMapR = ((baseline*focus)/(gtR+doffs)) / 1000     

# GT depth metrics
gtdminL = gtdMapL.min()
gtdminR = gtdMapR.min()
gtdmaxL = gtdMapL.max()
gtdmaxR = gtdMapR.max()
gtdvarL = gtdmaxL - gtdminL
gtdvarR = gtdmaxR - gtdminR

# Scaled GT depth metrics
gtsvarL = gtvarL*scale
gtsvarR = gtvarR*scale

# print("Ground truth depth left:\n")
# fig,axes = plt.subplots(1,1)
# axes.set_xlabel("X")
# axes.set_ylabel("Y")
# axes.set_title("gtdMapL")
# axes.imshow(gtdMapL,cmap='gray')
# #plt.colorbar('gray')
# plt.show()

# print("Ground truth depth right:\n")
# fig,axes = plt.subplots(1,1)
# axes.set_xlabel("X")
# axes.set_ylabel("Y")
# axes.set_title("gtdMapL")
# axes.imshow(gtdMapR,cmap='gray')
# plt.show()        

# Outputs before SGM calculation:
print("Size of original input image: %s x %s \n" % (width, height))
print("Sized of NN resized image: %s x %s \n" % (dpL.shape[1], dpL.shape[0]))
print("Mono-depth range estimation (left): %s [m] \n" % (dvarL))
print("Mono-depth range estimation (right): %s [m] \n" % (dvarR))

print("Mono-depth minimum disparity estimation (avg): %s [pix] \n" % (dMin))
print("Mono-depth minimum disparity estimation (avg): %s [pix] \n" % (dRange))

print("Ground truth minimum disparity (avg): %s [pix] \n" % (gtsmin))
print("Ground truch maximum disparity (avg): %s [pix] \n" % (gtsmax))

print("Ground truth disparity range (left): %s [pix] \n" % (gtvarL))
print("Ground truth disparity range (right): %s [pix] \n" % (gtvarL))
print("Scaled ground truth disparity range (left): %s [pix] \n" % (gtsvarL))
print("Scaled ground truth disparity range (right): %s [pix] \n" % (gtsvarL))   
print("Ground truth depth range (left): %s [m] \n" % (gtdvarL))
print("Ground truth depth range (right): %s [m] \n" % (gtdvarL)) 

# Block size for sum aggregation (good values: 5-9)
bS = 7 
bSf = np.float(bS)

# Rounded and conservatively underestimated dMin
dMin2 = int(np.round(dMin-(dMin/2)))

# Disparity range NOTE: For calculation 
#dR = np.arange(1,dRange)                     # Dynamic, debug here
dR = np.arange(dMin2, dRange)        # disparity 0 is ignored, non plausible value (inf distance))
dR = dR.astype(np.int16)

# If dMin starting above 1: offset must be added back to dispImg 
dD = dMin2-1

# Penalty terms
p1 = (0.5 * bSf * bSf)
p2 = (2 * bSf * bSf)

# Number of paths (SUPPORTS 1-8)
nP = 3

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

def costProc(d, pars, cIm, return_dict):
    
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
    if __name__ == '__main__':
        processes = []
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        for d in range(nP):        
            p = multiprocessing.Process(target = costProc, args = (d, pars, cIm, return_dict))
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

print("Calculating disparities in estimated range of %s pixels" % (dRange))

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

# Calculate raw cost
cIm = rawCost(imL, imR, bS, dR)

# Path search and cost aggregation
lIm = costAgg(cIm, p1, p2, nP)

# Sum across paths
S = np.sum(lIm, axis=3)

# Initial disparity map as disparity value at location of minimum cost across all paths:
dMap = np.argmin(S, axis=2) + dD

print("SGM Disparity map inital:\n")
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("SGM Disparity Image Initial")
axes.imshow(dMap,cmap='gray')
plt.show()

#Convert final disparities to .PFT for eval 
dMap2 = np.flipud(dMap)
dMap2 = dMap2.astype(np.float32)
filename = "dMap.pfm"
file = open(filename,"w")
dMap3 = save_pfm(file,dMap2, scale = 1)
file.close()

# Remove zero values for conversion to depth
dMap = dMap[np.isfinite(dMap)]
dMap = dMap.reshape(height2, width2)

# Depth map from disparity map
dpMap = np.zeros((height2, width2))
dpMap = ((baseline*focus2)/(dMap + doffs2)) / 1000

print("SGM Disparity map final:\n")
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("SGM Disparity Image Final")
axes.imshow(dMap,cmap='gray')
plt.show()

print("SGM Depth map:\n")
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("SGM Depth Image")
axes.imshow(dpMap,cmap='gray')
plt.show()

print("Time elapsed: --- %s seconds ---" % (time.time() - start))

imageio.imsave('dMap.png', dMap.astype(np.uint8))
imageio.imsave('dpMap.png', dpMap.astype(np.uint8))