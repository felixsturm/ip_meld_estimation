# -*- coding: utf-8 -*-
# search for max liver size to crop it later
import os
import pydicom as dcm

path = './dataset/dcms'
left,right,top,down = 1000,0,1000,0
maxH, maxW = 0,0

for folder in os.listdir(path):
    for file in os.listdir(os.path.join(path,folder)):
        try:
            ds = dcm.dcmread(os.path.join(path,folder,file))
        except dcm.errors.InvalidDicomError:
            continue

        image = ds.pixel_array
        s1 = sum(image).nonzero()
        s2 = sum(image.transpose()).nonzero()
        s1 = s1[0]
        s2 = s2[0]
        if left > s1[0]:
            left = s1[0]
        if right < s1[-1]:
            right = s1[-1]
        if top > s2[0]:
            top = s2[0]
        if down < s2[-1]:
            down = s2[-1]
        height = s2[-1] - s2[0]
        width = s1[-1] - s1[0]
        if maxH < height+1:
            maxH = height+1
        if maxW < width+1:
            maxW = width+1

ds = open("5_size.txt", "w")
ds.write(('maxH: %.0f')% maxH  + '\n' +
         ('maxW: %.0f') % maxW + '\n' +
         ('l: %.0f') % left + '\n' + 
         ('r: %.0f') % right + '\n' + 
         ('t: %.0f') % top + '\n' + 
         ('d: %.0f') % down) 
ds.close()
