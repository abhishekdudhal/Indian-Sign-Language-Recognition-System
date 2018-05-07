# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 23:17:00 2018
@author: abhi
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import plot_model
from keras import backend

backend.set_image_dim_ordering('th')
# input image dimensions
img_x, img_y = 200, 200
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
img_channels = 1
#number of classes 
no_classes = 10
#size of convolutional filter
no_conv = 3
#size of max pooling window
no_pool = 2
no_filters = [2,4,8,16,32,64,128,256,512]
dropout_ratio = [0,0.25,0.5,0.75,1]
input_shape = ( img_channels,img_x, img_y)

WeightFileName = ["adaptivethresholdmodeweight.hdf5","siftmodeweight.hdf5","nofiltermodeweight.hdf5"]

def createCNNModel(isBgModeOn):

    model = Sequential()

    model.add(Conv2D(no_filters[4], (no_conv, no_conv), padding='valid', activation='relu', input_shape=input_shape))
    model.add(Conv2D(no_filters[4], (no_conv, no_conv), activation='relu'))
    model.add(MaxPooling2D(pool_size=(no_pool, no_pool)))
    model.add(Dropout(dropout_ratio[1]))

    model.add(Conv2D(no_filters[5], (no_conv, no_conv), padding='valid', activation='relu'))
    model.add(Conv2D(no_filters[5], (no_conv, no_conv), activation='relu'))
    model.add(MaxPooling2D(pool_size=(no_pool, no_pool)))
    model.add(Dropout(dropout_ratio[1]))

    model.add(Conv2D(no_filters[6], (no_conv, no_conv), padding='valid', activation='relu'))
    model.add(Conv2D(no_filters[6], (no_conv, no_conv), activation='relu'))
    model.add(MaxPooling2D(pool_size=(no_pool, no_pool)))
    model.add(Dropout(dropout_ratio[1]))

    model.add(Conv2D(no_filters[7], (no_conv, no_conv), padding='valid', activation='relu'))
    model.add(Conv2D(no_filters[7], (no_conv, no_conv), activation='relu'))
    model.add(MaxPooling2D(pool_size=(no_pool, no_pool)))
    model.add(Dropout(dropout_ratio[1]))
    
    model.add(Conv2D(no_filters[8], (no_conv, no_conv), padding='valid', activation='relu'))
    model.add(Conv2D(no_filters[8], (no_conv, no_conv), activation='relu'))
    model.add(MaxPooling2D(pool_size=(no_pool, no_pool)))
    model.add(Dropout(dropout_ratio[1]))


    model.add(Flatten())
    model.add(Dense(no_filters[8], activation='relu'))
    model.add(Dropout(dropout_ratio[2]))
    model.add(Dense(no_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.get_config()

    plot_model(model, to_file='new_model.png', show_shapes = True)
    if(isBgModeOn>-1):
       wightFileName = WeightFileName[int(isBgModeOn)]
       print ("loading Weight File "+ str(wightFileName) + "...")
       model.load_weights(wightFileName)
    else:
        print("Creating New CNN for training...")
    return model

