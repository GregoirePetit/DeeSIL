#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Script for utility functions exploited by other programs

"""

import sys,os
import re
import numpy as np
from numpy.linalg import norm


def compute_dist_np(feat1, feat2):
    diff = feat1 - feat2
    feat_dist = np.dot(diff,diff)
    return feat_dist

def normalize_l2(v):
    '''
    L2-normalization of numpy arrays
    '''
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

