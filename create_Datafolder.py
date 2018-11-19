# -*- coding: utf-8 -*-
# copy files saved in txt file in folders according to their MELD-Scores
import os
from shutil import copyfile, rmtree

path = './dataset/dcms'
if os.path.exists(path):
    rmtree(path)    
counter = 1;
datalist = open("dataset.txt", "r")
for line in datalist:
    file,score = line.split("   ")
    try:
        if float(score) < 10.5:
            label = '1'
        elif float(score) < 15.5:
            label = '2'
        else:
            label = '3'
    except:
        continue

    if not os.path.exists(os.path.join(path,label)):
        os.makedirs(os.path.join(path,label))
    copyfile(file, os.path.join(path,label,str(counter)+'.dcm'))
    counter += 1;
datalist.close()   