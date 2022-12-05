#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import tqdm
from PIL import Image
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import torch
import glob


# In[ ]:


def box_save(path, img, model):
    coordinates = []
    results = model(img)
    coordinates.append(results.xyxy[0])
    return coordinates


# In[ ]:


def compute_intersect_area_rate_1(index, i):

    if i == 0:
        return 0
    
    if (classifications[0][i] != 1) and (classifications[0][i-1] != 1):
    
        x1, y1 = float(coordinates[index][i][0]), float(coordinates[index][i][1])
        x2, y2 = float(coordinates[index][i][2]), float(coordinates[index][i][3])
        x3, y3 = float(coordinates[index][i-1][0]), float(coordinates[index][i-1][1]) 
        x4, y4 = float(coordinates[index][i-1][2]), float(coordinates[index][i-1][3])

        if x2 < x3:
            intersect = 0

        if x1 > x4:
            intersect = 0

        if  y2 < y3:
              intersect = 0

        if  y1 > y4:
              intersect = 0

        left_up_x = max(x1, x3)
        left_up_y = max(y1, y3)
        right_down_x = min(x2, x4)
        right_down_y = min(y2, y4)

        width = right_down_x - left_up_x
        height =  right_down_y - left_up_y

        intersect = width * height

        area = (x2-x1)*(y2-y1)

        return intersect/area

    else:
        return 0


# In[2]:


def compute_intersect_area_rate_2(index, i):

    if (i == 0) or (i == 1):
        return 0
    
    if (classifications[0][i] != 1) and (classifications[0][i-2] != 1):
    
        x1, y1 = float(coordinates[index][i][0]), float(coordinates[index][i][1])
        x2, y2 = float(coordinates[index][i][2]), float(coordinates[index][i][3])
        x3, y3 = float(coordinates[index][i-2][0]), float(coordinates[index][i-2][1]) 
        x4, y4 = float(coordinates[index][i-2][2]), float(coordinates[index][i-2][3])

        if x2 < x3:
            intersect = 0

        if x1 > x4:
            intersect = 0

        if  y2 < y3:
              intersect = 0

        if  y1 > y4:
              intersect = 0

        left_up_x = max(x1, x3)
        left_up_y = max(y1, y3)
        right_down_x = min(x2, x4)
        right_down_y = min(y2, y4)

        width = right_down_x - left_up_x
        height =  right_down_y - left_up_y

        intersect = width * height

        area = (x2-x1)*(y2-y1)

        return intersect/area

    else:
        return 0


# In[ ]:


def compute_intersect_area_rate_3(index, i):

    if (i == 0) or (i == 1) or(i ==2):
        return 0
    
    if (classifications[0][i] != 1) and (classifications[0][i-3] != 1):
    
        x1, y1 = float(coordinates[index][i][0]), float(coordinates[index][i][1])
        x2, y2 = float(coordinates[index][i][2]), float(coordinates[index][i][3])
        x3, y3 = float(coordinates[index][i-3][0]), float(coordinates[index][i-3][1]) 
        x4, y4 = float(coordinates[index][i-3][2]), float(coordinates[index][i-3][3])

        if x2 < x3:
            intersect = 0

        if x1 > x4:
            intersect = 0

        if  y2 < y3:
              intersect = 0

        if  y1 > y4:
              intersect = 0

        left_up_x = max(x1, x3)
        left_up_y = max(y1, y3)
        right_down_x = min(x2, x4)
        right_down_y = min(y2, y4)

        width = right_down_x - left_up_x
        height =  right_down_y - left_up_y

        intersect = width * height

        area = (x2-x1)*(y2-y1)

        return intersect/area

    else:
        return 0


# In[ ]:


def compute_intersect_area_rate_4(index, i):

    if (i == 0) or (i == 1) or (i == 2) or (i == 3):
        return 0
    
    if (classifications[0][i] != 1) and (classifications[0][i-4] != 1):
    
        x1, y1 = float(coordinates[index][i][0]), float(coordinates[index][i][1])
        x2, y2 = float(coordinates[index][i][2]), float(coordinates[index][i][3])
        x3, y3 = float(coordinates[index][i-4][0]), float(coordinates[index][i-4][1]) 
        x4, y4 = float(coordinates[index][i-4][2]), float(coordinates[index][i-4][3])

        if x2 < x3:
            intersect = 0

        if x1 > x4:
            intersect = 0

        if  y2 < y3:
              intersect = 0

        if  y1 > y4:
              intersect = 0

        left_up_x = max(x1, x3)
        left_up_y = max(y1, y3)
        right_down_x = min(x2, x4)
        right_down_y = min(y2, y4)

        width = right_down_x - left_up_x
        height =  right_down_y - left_up_y

        intersect = width * height

        area = (x2-x1)*(y2-y1)

        return intersect/area

    else:
        return 0


# In[ ]:


def compute_intersect_area_rate_5(index, i):

    if (i == 0) or (i == 1) or (i == 2) or (i == 3) or (i == 4):
        return 0

    if (classifications[0][i] != 1) and (classifications[0][i-5] != 1):
    
        x1, y1 = float(coordinates[index][i][0]), float(coordinates[index][i][1])
        x2, y2 = float(coordinates[index][i][2]), float(coordinates[index][i][3])
        x3, y3 = float(coordinates[index][i-5][0]), float(coordinates[index][i-5][1]) 
        x4, y4 = float(coordinates[index][i-5][2]), float(coordinates[index][i-5][3])

        if x2 < x3:
            intersect = 0

        if x1 > x4:
            intersect = 0

        if  y2 < y3:
              intersect = 0

        if  y1 > y4:
              intersect = 0

        left_up_x = max(x1, x3)
        left_up_y = max(y1, y3)
        right_down_x = min(x2, x4)
        right_down_y = min(y2, y4)

        width = right_down_x - left_up_x
        height =  right_down_y - left_up_y

        intersect = width * height

        area = (x2-x1)*(y2-y1)

        return intersect/area

    else:
        return 0


# In[ ]:


def final_cost(path, img, classifications, coordinates):
    coordinates=coordinates.tolist()
    coordinates=torch.Tensor([coordinates])
    filling = 0
    fluoride = 0
    root_canal = 0

    for i, j in zip(range(len(coordinates[0])), range(len(classifications[0]))):

        a = float(coordinates[0][i][0]) #x1
        b = float(coordinates[0][i][1]) #y1
        c = float(coordinates[0][i][2]) #x2
        d = float(coordinates[0][i][3]) #y2

        if compute_intersect_area_rate_1(0, i) > 0.2:
            continue

        if compute_intersect_area_rate_2(0, i) > 0.2:
            continue

        if compute_intersect_area_rate_3(0, i) > 0.2:
            continue

        if compute_intersect_area_rate_4(0, i) > 0.2:
            continue

        if compute_intersect_area_rate_5(0, i) > 0.2:
            continue

        draw = ImageDraw.Draw(img)
        if classifications[0][j] == 0:
            draw.line([(a,b), (a,d), (c,d), (c,b), (a,b)], fill = "red", width=4)
            filling += 1

        if classifications[0][j] == 1:
            continue

        if classifications[0][j] == 2:
            draw.line([(a,b), (a,d), (c,d), (c,b), (a,b)], fill = "green", width=4)
            root_canal += 1

    cost = filling*25 + root_canal*55

    img.save((path+'/result/result.jpg'))
    img_r = Image.open(path+'/result/result.jpg')
    img_r = img_r.resize((1024, 400))

    ImageDraw.Draw(img_r).text((850, 335), "Filling : " + str(filling), fill = (255, 0, 0))
    ImageDraw.Draw(img_r).text((850, 350), "Root Canal : " + str(root_canal), fill = (0, 255, 0))
    ImageDraw.Draw(img_r).text((850, 365), "Cost : " + str(cost*10000), fill = (255, 255, 255))

    img_r.save((path+'/result/result.jpg'))
    return img_r

