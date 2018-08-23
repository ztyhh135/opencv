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
grid_size = int(sys.argv[3])
img_output = sys.argv[5]
#if_output_threshold=''
#if_output_threshold = sys.argv[5]
    
#img = cv.imread('DataSamples/ductile_iron0.jpg',0)
#img_input = 'DataSamples/ductile_iron0.jpg'
#grid_size = 10
img = cv.imread(img_input,0)
x,y = img.shape
img_splited = []
for i in range(grid_size):
#    if i==0:
#        img_splited.append(img[0:x,0:y/grid_size])
#    else:
    img_splited.append(img[0:x,y*i/grid_size:y*(i+1)/grid_size])
binary_image_splited = []
for i in range(grid_size):
    img_current = img_splited[i]
    threshold = 0
    otsu = 99999999
    for t in range(256):
        cluster_1 = np.where(img_current>t, img_current,0)
        cluster_2 = np.where(img_current<=t, img_current,0)
        cluster_2_fix = np.where(img_current<=t, img_current,-1)
        cluster_1_size = np.count_nonzero(cluster_1)
        if cluster_1_size > 0:
            cluster_1_mean = cluster_1.sum()/(cluster_1_size+.0)
            cluster_1_var = np.where(cluster_1>0, cluster_1,cluster_1_mean).var()*img_current.size/(cluster_1_size+.0)
        else:
            cluster_1_mean = 0
            cluster_1_var = 0
        if cluster_1_size < img_current.size:
            cluster_2_mean = cluster_2.sum()/(img_current.size - cluster_1_size+.0)
            cluster_2_var = np.where(cluster_2_fix!=-1,cluster_2_fix,cluster_2_mean).var()*img_current.size/(img_current.size - cluster_1_size+.0)
        else:
            cluster_2_mean = 0
            cluster_2_var=0
        
        
        otsu_min = (cluster_1_var*cluster_1_size + cluster_2_var*(img_current.size - cluster_1_size))/(img_current.size+.0)
        
        if otsu_min <= otsu:
            threshold = t
            otsu = otsu_min
    cluster_1 = np.where(img_current>=threshold, img_current,0)
    binary_image_splited.append(np.where(cluster_1==0,cluster_1,255))
    
binary_image = img
for i in range(grid_size):
#    if i==0:
#        binary_image[0:x,0:y/grid_size] = binary_image_splited[0]
#    else:
    binary_image[0:x,y*i/grid_size:y*(i+1)/grid_size] = binary_image_splited[i]
    
#if if_output_threshold=='--threshold':
#    print threshold
cv.imwrite(img_output, binary_image)
#plt.imshow(binary_image,cmap ='gray')
#cv.namedWindow("Image")   
#cv.imshow("Image", binary_image)   
#cv.waitKey (0)  
#cv.destroyAllWindows()