# -*- coding: utf-8 -*-
# reduce examples of class 1
# read Rescale Slope
# save data to numpy files
import os
import pydicom as dcm
import math
import numpy as np

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10

def sortKeyFunc(s):
    return int(os.path.basename(s)[:-4])

root = './dataset'
path = root + '/dcms'
h = 350
w = 400

size = 0
for folder in os.listdir(path):
    if int(folder) == 1:
        size += 42
    else:
        size += len(os.listdir(path+ '/' + folder))

X = np.zeros([int(size/3),h,w,3])
Y = np.zeros([int(size/3)])
Scales = np.zeros([int(size/3)])
Files = []
index = 0
# Main Loop
for folder in os.listdir(path):
    list = os.listdir(os.path.join(path,folder))
    list.sort(key=sortKeyFunc)
    tmpSc = np.zeros([300, 370, 3])
    st = -1
    # random start for class 1
    if int(folder) == 1:
        st = np.random.randint(0, (len(list) - 42)/3)*3
    for i in range(len(list)):
        # select only 42 of class 1
        if (i not in range(st,st+42)) and (st != -1):
            continue

        file = list[i]

        ds = dcm.dcmread(os.path.join(path,folder,file))

        scale = 1
        if "RealWorldValueMappingSequence" in ds:
            scale = ds.RealWorldValueMappingSequence[0].RealWorldValueSlope

        image = ds.pixel_array

        # find segment in image
        s1 = sum(image).nonzero()
        s2 = sum(image.transpose()).nonzero()
        s1 = s1[0]
        s2 = s2[0]

        # crop to segment
        img = image[s2[0]:s2[-1]+1,s1[0]:s1[-1]+1]
        height,width = img.shape
        startX = math.floor((w - width)/2)
        startY = math.floor((h - height)/2)
        endX = startX + width
        endY = startY + height

        # scale segment values
        tmpSc[startY:endY,startX:endX,np.mod(i,3)] = np.copy(img)

        if np.mod(i,3) == 2:
            X[index,:,:,:] = np.copy(tmpSc)
            Scales[index] = scale
            Y[index] = int(folder)
            tmpSc = np.zeros([h,w,3])
            Files = np.append(Files, list[i-2] +' '+ list[i-1] +' '+ list[i])
            index += 1

np.save('./dataset/X',X)
np.save('./dataset/Y',Y)
np.save('./dataset/Scales',Scales)
np.save('./dataset/Files',Files)