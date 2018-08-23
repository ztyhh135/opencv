# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:56:56 2018

@author: zty-pc2
"""

import cv2 as cv
import numpy as np
#from matplotlib import pyplot as plt
import sys


def findneighbours(neighbours, img, img_label, i, j):
    if i>0:
        if img[i-1,j]==0 and img_label[i-1,j]!=0:
            neighbours.append([i-1,j])
    if i<img.shape[0]-1:
        if img[i+1,j]==0 and img_label[i+1,j]!=0:
            neighbours.append([i+1,j])
    if j>0:
        if img[i,j-1]==0 and img_label[i,j-1]!=0:
            neighbours.append([i,j-1])
    if j<img.shape[1]-1:
        if img[i,j+1]==0 and img_label[i,j+1]!=0:
            neighbours.append([i,j+1])
    

img_input = sys.argv[2]
#grid_size = sys.argv[3]
size = int(sys.argv[4])
img_output=''
try:
    img_output = sys.argv[6]
except:
    IndexError
    
#if_output_threshold=''
#if_output_threshold = sys.argv[5]
    
#img = cv.imread('DataSamples/ductile_iron0.jpg',0)
#img_input = 'DataSamples/binary_15.png'
#grid_size = 10
#size = 500
img = cv.imread(img_input,0)
#img = cv.blur(img,(5,5))
#img = cv.imread(img_input,0)
x,y = img.shape

#neighbours=[]
linked = {}
img_label = np.zeros([x,y])
label = 1

for i in range(x):
    for j in range(y):
        if img[i,j]==0:
            neighbours=[]
            findneighbours(neighbours,img,img_label,i,j)
            if len(neighbours)==0:
                if not linked.has_key(label):
                    
                    linked[label] = [label]
                else:
                    if label not in linked[label]:
                        linked[label].append(label)
                img_label[i,j]=label
                label+=1
            else:
                l=[]
                for each in neighbours:
                    if img_label[each[0],each[1]]!=0:
                        l.append(img_label[each[0],each[1]])
                if len(l)>0:
                    img_label[i,j] = min(l)
                else:
                    img_label[i,j]
                for each in l:
                    if not linked.has_key(each):
                        linked[each] = [each]
                        for nl in l:
                            if nl not in linked[each]:
                                linked[each].append(nl)
                    else:
                        for nl in l:
                            if nl not in linked[each]:
                                linked[each].append(nl)
                                

for i in range(x):
    for j in range(y):
        if img[i,j]==0:
            while min(linked[img_label[i,j]])<img_label[i,j]:
                img_label[i,j] = min(linked[img_label[i,j]])
#for i in range(x):
#    for j in range(y):
#        if img[i,j]==0 and img_label[i,j]==0:
#            img_label[i,j]= label
#            findneighbours(neighbours, img, img_label, i, j)
#            while len(neighbours)>0:
#                current_x,current_y = neighbours.pop()
#                findneighbours(neighbours, img, img_label, current_x, current_y)
#                img_label[current_x,current_y] = label
#            label += 1
numbers = label-1
img_colored = np.zeros([x,y,3],dtype='uint8')
for i in range(x):
    for j in range(y):
        for k in range(3):
            img_colored[i,j,k]=255
single_color = np.arange(0,255,15)
colors = np.array([(i,j,k) for i in single_color for j in single_color for k in single_color])
colors_chosen = np.random.choice(colors.shape[0],label,replace=False)
for i in range(1,label):
    if np.sum(img_label==i)<=size:
        img_label[img_label==i]=0
        numbers -= 1
    else:
        pixels = np.sum(img_label==i)
#        print pixels, i
#        for each in img_colored[img_label==i]:
#            each = colors[colors_chosen[i]]
        img_colored[img_label==i]=[colors[colors_chosen[i]] for j in range(pixels)]
print numbers


#binary_image = np.where(cluster_1==0,cluster_1,255)
#if if_output_threshold=='--threshold':
#    print threshold
if img_output != '':
    cv.imwrite(img_output, img_colored)
#plt.imshow(binary_image,cmap ='gray')
#cv.namedWindow("Image")   
#cv.imshow("Image", img_colored)   
#cv.waitKey (0)  
#cv.destroyAllWindows()