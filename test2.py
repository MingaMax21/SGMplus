#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 20:05:37 2018

@author: max
"""

import numpy as np

mat = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,26,27]]])

first = mat[0,0,0]
last  = mat[2,2,2]

dMap = np.min(mat, axis=0)
dMap2 = np.argmin(mat,axis=0)
# Wanted: all ones

#Note: np.amin seems to equal np.min

# Matlab vs Python legend
# M3 = P0,  M2=P2, M1=P1
maam = np.amax(mat)