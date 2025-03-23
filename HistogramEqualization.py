# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:37:41 2019

@author: user
"""
import cv2
import numpy as np
import Basic_image_tools_v2 as bt
image=cv2.imread('D:/reset/mid/pollen.jpg',0)
image_d=image.astype(float)
image_N=np.zeros((256))
m,n =image.shape
image_eq=np.zeros((m,n))
pixel=np.zeros((256))
L_A=np.zeros((256))
L_A_B=np.zeros((256))
for i in range(m):
    for j in range(n):
        P_V=image[i,j]
        L_A[int(P_V)]=L_A[int(P_V)]+1
L_A_B[0]=L_A[0]
for i in range(1,256):
    L_A_B[i]=L_A_B[i-1]+L_A[i]        
L_A_B=((255)/(250000))*L_A_B##500*500pixels
S_array=L_A_B.copy()
#image_N=image_N.astype(np.uint8)  
for i in range(m):
    for j in range(n):
        pixel=image[i,j]
        image_eq[i,j]=L_A_B[pixel]
image_eq=image_eq.astype(np.uint8)

cv2.imshow('im',image)
cv2.imshow('eq',image_eq)

cv2.waitKey(0)
cv2.destroyAllWindows()
