#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 20:05:37 2018

@author: max
"""

import numpy as np

nP = 8
sD = 20 # displevels
# indexing
x_inds = np.array([[0],[3],[24],[32],[41],[15]])
y_inds = np.array([[1],[4],[25],[33],[42],[16]])


inds_in = np.where(np.logical_and(y_inds>=0, y_inds<=30))
inds_in = inds_in[0]

x_inds = x_inds[inds_in]
y_inds =y_inds[inds_in]

# repmatting (!= 80x1)
sL = x_inds.shape[0]
print(sL)

x_inds2 = np.kron(np.ones((sD,1)), x_inds)    # repmat(x_inds, 1, nP)
y_inds2 = np.kron(np.ones((sD,1)), y_inds)    # repmat(x_inds, 1, nP)

help1 = np.arange(0,sD)

z_inds2 = np.kron(np.ones((sL,1)), help1)

z_inds3 = z_inds2.ravel(1)

#z_inds3 = np.kron(np.ones((1,1)), z_inds2)