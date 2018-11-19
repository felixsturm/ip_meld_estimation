# -*- coding: utf-8 -*-
# normalizing and shuffling data
from tensorflow import keras as k
import numpy as np
import os

def norm(image, M):
    if M == 'mv':
        image = np.subtract(image,np.mean(image))
        image = np.divide(image, np.var(image))
    if M == 'n':
        image = np.divide(image,np.max(image))
    return image

root = '.' + os.sep + 'dataset'
X = np.load('.' + os.sep + 'dataset' + os.sep + 'X.npy')
Y = np.load('.' + os.sep + 'dataset' + os.sep + 'Y.npy')
Sc = np.load('.' + os.sep + 'dataset' + os.sep + 'Scales.npy')
Files = np.load('.' + os.sep + 'dataset' + os.sep + 'Files.npy')

normalization = 'n' # mv = zero mean + variance, n = norm tp range [0,1]
norm_about = 'e' # p = about patient, c = over class, e = else
v,idx = np.unique(Y, return_index=True)

# # scaling: if activated, comment out at begin
for i in range(X.shape[0]):
     X[i,:,:,:] = np.multiply(X[i,:,:,:],Sc[i])

if norm_about == 'p':
    # Main Loop
    for i in range(len(Y)):
        tmp = norm(X[i,:,:,:], normalization)
        X[i,:,:,:] = np.copy(tmp)

elif norm_about == 'c':
    X[0:idx[1],:,:,:] = np.copy(norm(X[0:idx[1],:,:,:], normalization))
    X[idx[1]:idx[2], :, :, :] = np.copy(norm(X[idx[1]:idx[2], :, :, :], normalization))
    X[idx[2]:, :, :, :] = np.copy(norm(X[idx[2]:, :, :, :], normalization))

else:
    X[0:idx[1], :, :, 0] = np.copy(norm(X[0:idx[1], :, :, 0], normalization))
    X[0:idx[1], :, :, 1] = np.copy(norm(X[0:idx[1], :, :, 1], normalization))
    X[0:idx[1], :, :, 2] = np.copy(norm(X[0:idx[1], :, :, 2], normalization))
    X[idx[1]:idx[2], :, :, 0] = np.copy(norm(X[idx[1]:idx[2], :, :, 0], normalization))
    X[idx[1]:idx[2], :, :, 1] = np.copy(norm(X[idx[1]:idx[2], :, :, 1], normalization))
    X[idx[1]:idx[2], :, :, 2] = np.copy(norm(X[idx[1]:idx[2], :, :, 2], normalization))
    X[idx[2]:, :, :, 0] = np.copy(norm(X[idx[2]:, :, :, 0], normalization))
    X[idx[2]:, :, :, 1] = np.copy(norm(X[idx[2]:, :, :, 1], normalization))
    X[idx[2]:, :, :, 2] = np.copy(norm(X[idx[2]:, :, :, 2], normalization))

p = np.load(os.path.join(root,'permutation.npy'))
x = [X[i,:,:,:] for i in p]
y = [Y for i in p]

y = np.asarray(y)
x = np.asarray(x)

np.save(os.path.join(root,'k_dataX_' + normalization + '_' + norm_about), x)
np.save(os.path.join(root,'k_dataY_'+ normalization + '_' + norm_about), y)
np.save(os.path.join(root,'k_datafiles_' + normalization + '_' + norm_about), files)