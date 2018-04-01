# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 23:26:16 2018
@author: abhi
"""
import cnnModel
from keras import backend
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
from keras.utils import np_utils
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

backend.set_image_dim_ordering('th')
model = None
path = ""
no_epoch = 50
batch_size = 64
    
def trainModel():
    global model,path
    model = cnnModel.createCNNModel(-1)
    no_classes = cnnModel.no_classes
    img_x = cnnModel.img_x
    img_y = cnnModel.img_y
    img_channels = cnnModel.img_channels
    dataSetType = int(input("Enter 1 for Adaptive Threshold Mode Dataset \nEnter 2 for Sift Mode Dataset \nEnter 3 for No Filter Mode Dataset\n"))
    if dataSetType == 1:
        path = './AdaptiveThresholdModeDataSet'
    elif dataSetType == 2:
        path = './SiftModeDataSet'
    elif dataSetType == 3:
        path = './NoFilterModeDataSet'
        
    #create dataset array   
    listing = os.listdir(path)
    listing.sort()
    dataset = []
    for name in listing:
        dataset.append(name)
        
    image = np.array(Image.open(path +'/' + dataset[0]))
    #fnd size of image
    m,n = image.shape[0:2]
    #find dataset size
    dataset_size = len(dataset)
    # create matrix to store all flattened images
    image_matrix = np.array([np.array(Image.open(path+ '/' + images).convert('L')).flatten() for images in dataset], dtype = 'f')
    label=np.ones((dataset_size,),dtype = int)
    samples_per_class = dataset_size / no_classes
    s = 0
    r = samples_per_class
    for classIdentifier in range(no_classes):
        label[int(s):int(r)] = classIdentifier
        s = r
        r = s + samples_per_class  
    data,Label = shuffle(image_matrix,label, random_state=2)
    train_data = [data,Label]
    (X, y) = (train_data[0],train_data[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
    X_train = X_train.reshape(X_train.shape[0], img_channels, img_x, img_y)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_x, img_y)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = np_utils.to_categorical(y_train, no_classes)
    Y_test = np_utils.to_categorical(y_test, no_classes)
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=no_epoch,verbose=1, validation_split=0.25) 
    
    print(history.history.keys())
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc.png')
    plt.close('all')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.close('all')
    print("Model trained successfully")
    
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    if dataSetType == 1:
             model.save_weights("./adaptivethresholdmodeweight.hdf5",overwrite=True)
    elif dataSetType == 2:
             model.save_weights("./siftmodeweight.hdf5",overwrite=True)
    elif dataSetType == 3:
             model.save_weights("./nofiltermodeweight.hdf5",overwrite=True)
   