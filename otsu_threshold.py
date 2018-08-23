# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:56:56 2018

@author: zty-pc2
"""

import cv2 as cv
import numpy as np
#from matplotlib import pyplot as plt
import sys

img_input = sys.argv[2]
img_output = sys.argv[4]
if_output_threshold=''
if_output_threshold = sys.argv[5]
    
#img = cv.imread('DataSamples/ductile_iron0.jpg',0)
#img_input = 'DataSamples/ductile_iron0.jpg'
img = cv.imread(img_input,0)
x,y = img.shape
threshold = 0
otsu = 99999999
for t in range(256):
    cluster_1 = np.where(img>t, img,0)
    cluster_2 = np.where(img<=t, img,0)
    cluster_2_fix = np.where(img<=t, img,-1)
    cluster_1_size = np.count_nonzero(cluster_1)
    if cluster_1_size > 0:
        cluster_1_mean = cluster_1.sum()/(cluster_1_size+.0)
        cluster_1_var = np.where(cluster_1>0, cluster_1,cluster_1_mean).var()*img.size/(cluster_1_size+.0)
    else:
        cluster_1_mean = 0
        cluster_1_var = 0
    if cluster_1_size < img.size:
        cluster_2_mean = cluster_2.sum()/(img.size - cluster_1_size+.0)
        cluster_2_var = np.where(cluster_2_fix!=-1,cluster_2_fix,cluster_2_mean).var()*img.size/(img.size - cluster_1_size+.0)
    else:
        cluster_2_mean = 0
        cluster_2_var=0
    
    
    otsu_min = (cluster_1_var*cluster_1_size + cluster_2_var*(img.size - cluster_1_size))/(img.size+.0)
    
    if otsu_min <= otsu:
        threshold = t
        otsu = otsu_min
cluster_1 = np.where(img>=threshold, img,0)
binary_image = np.where(cluster_1==0,cluster_1,255)
if if_output_threshold=='--threshold':
    print threshold
cv.imwrite(img_output, binary_image)
#plt.imshow(binary_image,0)