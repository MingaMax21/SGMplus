#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 12:30:39 2018

@author: max
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from skimage import img_as_ubyte
from skimage import transform
from scipy import signal
from scipy import misc
import time

# Start timer
start = time.time()

# Read Image
imL = color.rgb2gray(io.imread("./data/mtlb_ex1/imL.png"))      # ADAPT PATH
imR = color.rgb2gray(io.imread("./data/mtlb_ex1/imR.png"))      # ADAPT PATH
h,w = imL.shape
w2 = 360
h2 = np.int(w2*(h/w))
s2 = (h2,w2)
imL = transform.resize(imL,s2)
imR = transform.resize(imR,s2)

imL = img_as_ubyte(imL)
imR = img_as_ubyte(imR)
imL = imL.astype(np.int16)    # Dumb workaround to make integer subtraction on images possible (TODO!)
imR = imR.astype(np.int16)

"""
The following image sizes occur in local middlebury seta:
Mask: 450 x 375             (1.2)
Motorcycle: 2964 x 2000     (1.2)
Mtlb: 384 x 288             (1.3_)
Piano: 2820 x 1920          (1.44875)
Recycle: 2864 x 1924        (1.4886..)
Umbrella: 2960 x 2016       (1.4683..)

Clearly no standard aspect ratio exists. Solution will be to force width to 360 and scale height 
"""
# Print original image dims and display image
fig,axes = plt.subplots(1,1)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_title("imL")
axes.imshow(imL,cmap='gray')
plt.show()
print(imL.shape)

# Downsample Image (set max dim to 360, scale height accordingly)
# h,w = imL.shape
# w2 = 360
# h2 = np.int(w2*(h/w))
# s2 = (h2,w2)
# imNew = transform.resize(imL,s2)
# imNew = img_as_ubyte(imNew)
# imNew = imNew.astype(np.int16)

