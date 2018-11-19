# -*- coding: utf-8 -*-
# create network an run training and test
# saves best models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks, Sequential
from tensorflow.keras import activations as acts
from tensorflow.keras.layers import Dense,Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D,  Dropout
from sklearn.model_selection import train_test_split as tt_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os
import numpy as np

# load Data
X = np.load('.' + os.sep + 'X.npy')
Y = np.load('.' + os.sep + 'Y.npy')

# Part 2 - Fitting the model to the images
datagen = ImageDataGenerator(shear_range=0.05,
                             zoom_range=0.05,
                             height_shift_range=5,
                             width_shift_range=5,
                             rotation_range=25,
                             horizontal_flip=True,
                             fill_mode='nearest',
                             data_format="channels_last")
epoch = 10
k = 5

# set destination folder
path = '.' + os.sep + 'results' + os.sep + '3x8_3x16'
if not os.path.exists(path):
    os.mkdir(path)
for i in range(k):
    if not os.path.exists(path + os.sep + str(i)):
        os.mkdir(path + os.sep + str(i))

accs = np.zeros((k,epoch))
val_accs = np.zeros((k,epoch))
loss = np.zeros((k,epoch))
val_loss = np.zeros((k,epoch))
kf = KFold(n_splits=k)

p = np.load('./permutation.npy')
X = [X[i,:,:,:] for i in p]
Y = [Y[i] for i in p]

Y = np.asarray(Y)
X = np.asarray(X)
i=0
for train_index, test_index in kf.split(X):
    x_train, x_test = X[train_index,:,:,:], X[test_index,:,:,:]
    y_train, y_test = Y[train_index], Y[test_index]

    # create model
    model = Sequential()
    # Step 1 - Convolution
    model.add(Conv2D(16, (3, 3), input_shape=(350, 400, 3), activation=acts.relu))
    model.add(Conv2D(16, (3, 3), activation=acts.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation=acts.relu))
    model.add(Conv2D(8, (3, 3), activation=acts.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (3, 3), activation=acts.relu))
    model.add(Conv2D(8, (3, 3), activation=acts.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # classificator
    model.add(Flatten())
    model.add(Dense(units=128, activation=acts.relu))
    model.add(Dense(units=64, activation=acts.relu))
    model.add(Dense(units=32, activation=acts.relu))
    model.add(Dense(units=16, activation=acts.relu))
    model.add(Dense(units=3, activation=acts.softmax))

    # Compiling the CNN
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if i == 0:
        model.summary()

    # set callbacks
    mcloss_cb = callbacks.ModelCheckpoint(filepath=path + os.sep + str(i) + os.sep + 'lowValLoss.hdf5', monitor='val_loss', save_best_only=True)
    mcacc_cb = callbacks.ModelCheckpoint(filepath=path + os.sep + str(i) + os.sep + 'bestAcc.hdf5', monitor='val_acc', save_best_only=True)
    esloss_cb = callbacks.EarlyStopping(patience=1)

    cbs = []
    cbs.append(mcloss_cb)
    cbs.append(mcacc_cb)
    cbs.append(esloss_cb)

    # fits the model on batches with real-time data augmentation:
    his = model.fit_generator(datagen.flow(x_train, y_train, batch_size=8, save_to_dir=path + os.sep + str(i) + os.sep, save_prefix='train'), validation_data=(x_test,y_test),
                              callbacks= cbs, steps_per_epoch=100, epochs=epoch, verbose=1)

    accs[i,0:len(his.history['acc'])] = his.history['acc']
    val_accs[i,0:len(his.history['val_acc'])] = his.history['val_acc']
    loss[i,0:len(his.history['loss'])] = his.history['loss']
    val_loss[i,0:len(his.history['val_loss'])] = his.history['val_loss']

    np.save(path + os.sep + 'acc', accs)
    np.save(path + os.sep + 'val_acc', val_accs)
    np.save(path + os.sep + 'val_loss', val_loss)
    np.save(path + os.sep + 'loss', loss)
    i += 1


np.save(path + os.sep + 'acc', accs)
np.save(path + os.sep + 'val_acc', val_accs)
np.save(path + os.sep + 'val_loss', val_loss)
np.save(path + os.sep + 'loss', loss)


# list all data in history
print(his.history.keys())
# summarize history for accuracy
plt.plot(np.mean(accs, axis=0))
plt.plot(np.mean(val_accs, axis=0))
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(np.mean(loss, axis=0))
plt.plot(np.mean(val_loss, axis=0))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()