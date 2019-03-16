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
from struct import unpack

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

gt = readGT("./dMap.pfm") 

# Plot pfm disparities
print("Ground truth disp left:\n")
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("gtL")
axes.imshow(gt,cmap='gray')
plt.show()


