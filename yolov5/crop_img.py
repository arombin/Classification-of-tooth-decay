#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
import torch
import yaml
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm
import glob


# In[ ]:


def crop(path, img_list, model):

    coordinates = []
    for i in range (len(img_list)):
        im = img_list[i]
        results = model(im)
        coordinates.append(results.xyxy[0])
        
    for i in range(len(coordinates)):
        coordinates[i] = coordinates[i].cpu().numpy() 
    
    for i in tqdm(range(len(img_list))):
        img = Image.open(img_list[i])
        for j in range(len(coordinates[i])):

            a = float(coordinates[i][j][0]) #x1
            b = float(coordinates[i][j][1]) #y1
            c = float(coordinates[i][j][2]) #x2
            d = float(coordinates[i][j][3]) #y2

            box = (a, b, c, d)
            img_c = img.crop(box)
            img_c.save(r''+path+'/yolov5/data/teethCrop/' + str(i) + '_' + str(j) + '.jpg')

