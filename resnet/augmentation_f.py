#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os 
import cv2
from PIL import Image
from numpy import expand_dims
import tensorflow as tf


# In[2]:


def flip(classname, path1, path2):
    path = path1+'/'+classname+'/'
    os.chdir(path) 
    files = os.listdir(path) #파일이름 list로 받아줘영

    for file in files: #상하로 flip하고 저장
        path = path1+'/'+classname+'/'
        os.chdir(path)
        data = cv2.imread(file)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        image = cv2.flip(data, 0)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        save_file = path2+'/'+classname+'/'
        os.chdir(save_file)
        name = 'flip0_'+file
        cv2.imwrite(name, image)

    path = path1+'/'+classname+'/'
    os.chdir(path) 
    files = os.listdir(path) #파일이름 list로 받아줘영

    for file in files: #좌우로 flip하고 저장
        path = path1+'/'+classname+'/'
        os.chdir(path)
        data = cv2.imread(file)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        image = cv2.flip(data, 1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        save_file = path2+'/'+classname+'/'
        os.chdir(save_file)
        name = 'flip1_'+file
        cv2.imwrite(name, image)
        
    path = path1+'/'+classname+'/'
    os.chdir(path) 
    files = os.listdir(path) #파일이름 list로 받아줘영
        
    for file in files: #원본 데이터 저장
        path = path1+'/'+classname+'/'
        os.chdir(path)
        data = cv2.imread(file)
        save_file = path2+'/'+classname+'/'
        os.chdir(save_file)
        cv2.imwrite(file, data)
        

# In[3]:


def augmentation(path):
    path_in = path+'/Team1/resnet/data/labeled'
    path_out = path+'/Team1/resnet/data/augmentation'
    os.chdir(path_in) 
    files = os.listdir(path_in)
    
    for name in files:
        flip(name, path_in, path_out)

