#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:07:11 2019

@author: ssw
"""
import cv2
import os
from PIL import Image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from albumentations import RandomContrast, RandomGamma, CLAHE

image_dir = '/home/gpuserver/competition/Tianzhi2/rgb/data4/src'
out_dir = './rgb8/data4/src'

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

imgPath_list = glob(os.path.join(image_dir, '*.tif'))

for imgPath in imgPath_list:
    print('Dealing with image:', imgPath)
    imgName = os.path.split(imgPath)[-1]
    img = np.array(Image.open(imgPath))
    
    if len(img.shape) == 2:
        '''
        img = np.array([img,img,img]).transpose(1,2,0)
        filter_image = CLAHE(clip_limit=(1.0,2.0),tile_grid_size=(3, 3),always_apply=True)(image=img)['image']
        Image.fromarray(filter_image.astype(np.uint8)).save(os.path.join(out_dir,'clahe_'+imgName))        
        filter_image = cv2.ximgproc.guidedFilter(filter_image, filter_image, 32, (255*0.2)**2)
        '''
        continue
        
    else:
        filter_image = cv2.ximgproc.guidedFilter(img, img, 8, (255*0.1)**2) #0.2^2;0.4^2;0.1^2
    Image.fromarray(filter_image.astype(np.uint8)).save(os.path.join(out_dir,imgName))
