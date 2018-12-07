#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 20:05:37 2018

@author: max
"""

import numpy as np

mat = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,27,27]]])

miim = np.amin(mat)

maam = np.amax(mat)